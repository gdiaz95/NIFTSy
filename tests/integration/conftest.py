class FakeLLMBackend:
    def generate_batch(self, prompts, config=None):
        return [f"synthetic text for prompt {i}" for i in range(len(prompts))]
