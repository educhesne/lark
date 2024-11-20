from copy import deepcopy, copy
from typing import Dict, Any, Generic, List, Optional
from collections.abc import Mapping
import re
from ast import Expression as AstExpression, Module as AstModule, fix_missing_locations
from ..lexer import Token, LexerThread
from ..common import ParserCallbacks

from .lalr_analysis import Shift, Reduce, ParseTableBase, StateT
from lark.exceptions import UnexpectedToken

###{standalone

class ParseConf(Generic[StateT]):
    __slots__ = 'parse_table', 'callbacks', 'start', 'start_state', 'end_state', 'states', 'python_header'

    parse_table: ParseTableBase[StateT]
    callbacks: ParserCallbacks
    start: str
    python_header: Optional[AstModule]

    start_state: StateT
    end_state: StateT
    states: Dict[StateT, Dict[str, tuple]]

    def __init__(self, parse_table: ParseTableBase[StateT], callbacks: ParserCallbacks, start: str, python_header: Optional[AstModule]):
        self.parse_table = parse_table

        self.start_state = self.parse_table.start_states[start]
        self.end_state = self.parse_table.end_states[start]
        self.states = self.parse_table.states

        self.callbacks = callbacks
        self.start = start
        self.python_header = python_header


class GlobalVariables():
    "Basic class to collect global variables for the attribute ast evaluation"
    pass


def eval_attribute(ast: Optional[AstExpression], stack: list, GLOBAL: GlobalVariables, header: Optional[AstModule]) -> Any:
    # evaluates the ast in a given context; as much as possible prevents side effects by allowing the reference to
    # only two external variables: GLOBAL, an instance of GlobalVariables, and stack, the current stack of attributes
    globals_dict = dict()
    locals_dict = dict()
    if ast:
        if header:
            exec(compile(header, filename="<ast>", mode="exec"), globals_dict, locals_dict)
            assert 'GLOBAL' not in locals_dict and 'stack' not in locals_dict, "GLOBAL and stack are reserved variables"
        # the expression are evaluated in a local context where "stack" and "GLOBAL" are defined
        locals_dict.update({'GLOBAL': GLOBAL, 'stack': stack})
        return eval(compile(fix_missing_locations(ast), filename="<ast>", mode="eval"), globals_dict, locals_dict)
    else:
        return None


class ContextualTransitions(Mapping[Token, tuple]):
    # wrapper class for the transitions of the automaton
    # when a lookahead is consulted, an optional contextual terminal ast is evaluated in the current state
    # of the parser; it returns a regex which is matched against the value of the lookahead token
    # to determine if the transition is admissible
    # so the transition mapping now takes token as keys, and the existence of an action depends on the token value, not only its type
    # (this all needs to be redesigned)

    __slots__ = ("states", "transitions", "state_stack", "attribute_stack", "global_vars", "python_header", "ctx_term")

    states: Dict
    transitions: Dict[str, tuple]
    state_stack: list
    attribute_stack: list
    global_vars: GlobalVariables
    python_header: Optional[AstModule]
    ctx_term: bool

    def __init__(self, transitions: Dict[str, tuple], states: Dict[StateT, Dict[str, tuple]], state_stack: list, attribute_stack: list, 
                 global_vars: GlobalVariables, python_header: Optional[AstModule], ctx_term:bool=True):
        self.states = states
        self.transitions = transitions
        self.state_stack = state_stack
        self.attribute_stack = attribute_stack
        self.global_vars = global_vars
        self.python_header = python_header or None
        self.ctx_term = ctx_term

    def __getitem__(self, token: Token):
        # check the token value against the lookahead pattern only if next action is Shift
        # otherwise the attribute stack is not yet ready for the evaluation of the contextual terminal ast
        # (avoids evaluating twice the attributes in the calls to feed_token)
        action, _, ast_pattern = self.transitions[token.type]
        if action is Reduce or ast_pattern is None:
            return self.transitions[token.type]

        pattern = self.lookahead_pattern(token.type)
        if pattern and (not self.ctx_term or re.match(pattern, token.value)):
            return self.transitions[token.type]
        else:
            raise KeyError

    def __iter__(self):
        for key in self.transitions.__iter__():
            pattern = self.lookahead_pattern(key)
            if pattern:
                yield Token(key, pattern)
            else:
                continue

    def __len__(self):
        return self.transitions.__len__()

    def lookahead_pattern(self, token_type: str):
        # evaluation of the lookahead ast; simulates Reduce actions if necessary
        # and returns a regex
        action, rule, ast_pattern = self.transitions[token_type]
        if ast_pattern is None:
            # match anything
            return '(.*?)'

        state_stack, attribute_stack, global_vars = deepcopy(self.state_stack), deepcopy(self.attribute_stack), deepcopy(self.global_vars)
        # the ast_pattern is meant to be evaluated on an attribute stack after all the reduce actions
        # have occurred
        # (this is redundant with feed_token; needs refacto)
        while action != Shift:
            size = len(rule.expansion)
            attribute = eval_attribute(rule.ast, attribute_stack, global_vars, self.python_header)
            if size:
                del state_stack[-size:]
                del attribute_stack[-size:]

            _action, new_state, _ = self.states[state_stack[-1]][rule.origin.name]
            assert _action is Shift
            state_stack.append(new_state)
            attribute_stack.append(attribute)

            try:
                action, rule, _ = self.states[new_state][token_type]
            except KeyError:
                return ''

        pattern = eval_attribute(ast_pattern, attribute_stack, global_vars, self.python_header)
        return pattern


def contextual_states(states: Dict[StateT, Dict[str, tuple]], state_stack: list, attribute_stack: list, global_vars: GlobalVariables,
                       python_header: Optional[AstModule], ctx_term:bool=True) -> Mapping[StateT, Mapping[Token, tuple]]:
    return {k: ContextualTransitions(v, states, state_stack, attribute_stack, global_vars, python_header, ctx_term) for k, v in states.items()}


class ParserState(Generic[StateT]):
    __slots__ = 'parse_conf', 'lexer', 'state_stack', 'value_stack', 'attribute_stack', 'python_header', 'global_vars'

    parse_conf: ParseConf[StateT]
    lexer: LexerThread
    state_stack: List[StateT]
    value_stack: list
    attribute_stack: list
    python_header: Optional[AstModule]
    global_vars: GlobalVariables

    def __init__(self, parse_conf: ParseConf[StateT], lexer: LexerThread, state_stack=None, value_stack=None, 
                 attribute_stack=None, global_vars=None):
        self.parse_conf = parse_conf
        self.lexer = lexer
        self.state_stack = state_stack or [self.parse_conf.start_state]
        self.value_stack = value_stack or []
        self.attribute_stack = attribute_stack or []
        self.python_header = parse_conf.python_header
        self.global_vars = global_vars or GlobalVariables()

    @property
    def position(self) -> StateT:
        return self.state_stack[-1]

    # Necessary for match_examples() to work
    def __eq__(self, other) -> bool:
        if not isinstance(other, ParserState):
            return NotImplemented
        return len(self.state_stack) == len(other.state_stack) and self.position == other.position

    def __copy__(self):
        return self.copy()

    def copy(self, deepcopy_values=True) -> 'ParserState[StateT]':
        return type(self)(
            self.parse_conf,
            self.lexer, # XXX copy
            copy(self.state_stack),
            deepcopy(self.value_stack) if deepcopy_values else copy(self.value_stack),
            deepcopy(self.attribute_stack),
            deepcopy(self.global_vars)
        )

    def feed_token(self, token: Token, is_end=False, ctx_term=True) -> Any:
        state_stack = self.state_stack
        value_stack = self.value_stack
        attribute_stack = self.attribute_stack     # the stack of synthesized attributes
        end_state = self.parse_conf.end_state
        callbacks = self.parse_conf.callbacks
        python_header = self.python_header
        global_vars = self.global_vars
        states = contextual_states(self.parse_conf.states, state_stack, attribute_stack, global_vars, python_header, ctx_term)


        while True:
            state = state_stack[-1]
            try:
                action, arg, _ = states[state][token]
            except KeyError:
                expected = {s for s in states[state].keys() if s.type.isupper()}
                raise UnexpectedToken(token, expected, state=self, interactive_parser=None)

            assert arg != end_state

            if action is Shift:
                # shift once and return
                assert not is_end
                state_stack.append(arg)
                value_stack.append(token if token.type not in callbacks else callbacks[token.type](token))
                attribute_stack.append(token.value)           # the attribute of a token is its value
                return
            else:
                # reduce+shift as many times as necessary
                rule = arg
                size = len(rule.expansion)

                # the synthesized attribute of a non-terminal symbol is the evaluation of the expression 
                # attached to the rule deriving it
                attribute = eval_attribute(rule.ast, attribute_stack, global_vars, python_header)

                if size:
                    s = value_stack[-size:]
                    del state_stack[-size:]
                    del value_stack[-size:]
                    del attribute_stack[-size:]
                else:
                    s = []

                value = callbacks[rule](s) if callbacks else s

                _action, new_state, _ = states[state_stack[-1]][Token(rule.origin.name, "")]
                assert _action is Shift
                state_stack.append(new_state)
                value_stack.append(value)
                attribute_stack.append(attribute)

                if is_end and state_stack[-1] == end_state:
                    return value_stack[-1], attribute_stack[-1]
###}
