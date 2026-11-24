class StageUnresolved(RuntimeError):
    """A method-level failure with valid evidence, not an infrastructure error."""
    def __init__(self,stage,reason):
        super().__init__(reason)
        self.stage=stage


class CandidateInvalid(ValueError):
    """A measured/syntactically invalid candidate, not a campaign failure."""
    def __init__(self, reason, evidence=None):
        super().__init__(reason)
        self.evidence = evidence or {}
