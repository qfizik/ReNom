import numpy as np
import renom as rm


class graph_container:

    def __init__(self):
        self.operations = set()
        self.roles = set()
        self.roles_ops = dict()

    def add(self, call_dict):
        for call in call_dict:
            call.setup()
            self.operations.add(call)
        for op in self.operations:
            for role in op._op.roles:
                if role not in self.roles_ops:
                    self.roles_ops[role] = set()
                self.roles_ops[role].add(op)
                self.roles.add(role)

    def feed(self, to_replace, replace_with):
        for placeholder in self.roles_ops['placeholder']:
            if placeholder.identifier == id(to_replace):
                placeholder.link(replace_with)
