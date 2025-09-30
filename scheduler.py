class AIAYNScheduler:
    """
    - lr scheduler from "Attention Is All You Need" Paper (AIAYN)
    """

    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self):
        lr = self.d_model**-0.5 * min(
            self.step_num**-0.5, self.step_num * (self.warmup_steps**-1.5)
        )

        return lr

    def get_lr(self):
        return self._get_lr()
