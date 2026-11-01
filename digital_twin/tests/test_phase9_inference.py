import unittest
from unittest.mock import patch

import torch

from ai.inference import NegotiationInference


class FakeTokenizer:
    def __init__(self, output):
        self.output = output
        self.last_prompt = None

    def __call__(self, prompt, **kwargs):
        self.last_prompt = prompt
        return {
            "input_ids": torch.tensor([[1, 2, 3]])
        }

    def decode(self, generated, skip_special_tokens=True):
        return self.output


class FakeModel:
    def __init__(self):
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[1, 2, 3]])


class NegotiationInferenceTests(unittest.TestCase):

    OWN_INPUT = "1 4 4 1 1 2"
    PARTNER_INPUT = "1 4 4 1 1 2"
    DIALOGUE = "I propose a fair allocation."

    VALID_ALLOCATION = (
        "item0=1 item1=0 item2=1 "
        "item0=0 item1=4 item2=0"
    )

    def create_inference(self, output):
        tokenizer = FakeTokenizer(output)
        model = FakeModel()

        with patch(
            "ai.inference.load_negotiation_model",
            return_value=(tokenizer, model),
        ):
            inference = NegotiationInference()

        return inference, tokenizer, model

    def test_valid_allocation_prediction(self):
        inference, _, _ = self.create_inference(
            self.VALID_ALLOCATION
        )

        result = inference.predict(
            self.OWN_INPUT,
            self.PARTNER_INPUT,
            self.DIALOGUE,
        )

        self.assertEqual(result.category, "allocation")
        self.assertTrue(result.is_supported)
        self.assertTrue(result.allocation_valid)
        self.assertTrue(result.is_acceptable)
        self.assertEqual(result.validation_reason, "valid")

    def test_invalid_allocation_is_rejected(self):
        invalid_output = (
            "item0=2 item1=0 item2=1 "
            "item0=0 item1=4 item2=0"
        )

        inference, _, _ = self.create_inference(invalid_output)

        result = inference.predict(
            self.OWN_INPUT,
            self.PARTNER_INPUT,
            self.DIALOGUE,
        )

        self.assertEqual(result.category, "allocation")
        self.assertTrue(result.is_supported)
        self.assertFalse(result.allocation_valid)
        self.assertFalse(result.is_acceptable)
        self.assertEqual(
            result.validation_reason,
            "allocation_does_not_conserve_items",
        )

    def test_negative_allocation_is_rejected(self):
        negative_output = (
            "item0=-1 item1=0 item2=1 "
            "item0=2 item1=4 item2=0"
        )

        inference, _, _ = self.create_inference(negative_output)

        result = inference.predict(
            self.OWN_INPUT,
            self.PARTNER_INPUT,
            self.DIALOGUE,
        )

        self.assertFalse(result.allocation_valid)
        self.assertFalse(result.is_acceptable)
        self.assertEqual(
            result.validation_reason,
            "negative_allocation_value",
        )

    def test_unsupported_output_is_rejected(self):
        inference, _, _ = self.create_inference(
            "This is an unrelated response."
        )

        result = inference.predict(
            self.OWN_INPUT,
            self.PARTNER_INPUT,
            self.DIALOGUE,
        )

        self.assertEqual(result.category, "other")
        self.assertFalse(result.is_supported)
        self.assertFalse(result.is_acceptable)
        self.assertEqual(
            result.validation_reason,
            "unsupported_output_format",
        )

    def test_generation_uses_deterministic_settings(self):
        inference, tokenizer, model = self.create_inference(
            self.VALID_ALLOCATION
        )

        inference.predict(
            self.OWN_INPUT,
            self.PARTNER_INPUT,
            self.DIALOGUE,
        )

        self.assertIsNotNone(tokenizer.last_prompt)
        self.assertEqual(len(model.generate_calls), 1)

        generation_args = model.generate_calls[0]

        self.assertEqual(generation_args["max_new_tokens"], 128)
        self.assertEqual(generation_args["num_beams"], 1)
        self.assertFalse(generation_args["do_sample"])

    def test_invalid_generation_configuration_is_rejected(self):
        with patch("ai.inference.load_negotiation_model") as loader:

            with self.assertRaises(ValueError):
                NegotiationInference(max_input_length=0)

            with self.assertRaises(ValueError):
                NegotiationInference(max_new_tokens=0)

            with self.assertRaises(ValueError):
                NegotiationInference(num_beams=0)

            loader.assert_not_called()

    def test_disagreement_is_supported_but_not_acceptable(self):
        inference, _, _ = self.create_inference("<disagree>")

        result = inference.predict(
            self.OWN_INPUT,
            self.PARTNER_INPUT,
            self.DIALOGUE,
        )

        self.assertTrue(result.is_supported)
        self.assertFalse(result.is_acceptable)
        self.assertEqual(result.category, "disagreement")

    def test_no_agreement_is_supported_but_not_acceptable(self):
        inference, _, _ = self.create_inference("<no_agreement>")

        result = inference.predict(
            self.OWN_INPUT,
            self.PARTNER_INPUT,
            self.DIALOGUE,
        )

        self.assertTrue(result.is_supported)
        self.assertFalse(result.is_acceptable)
        self.assertEqual(result.category, "no_agreement")

    def test_disconnect_is_supported_but_not_acceptable(self):
        inference, _, _ = self.create_inference("<disconnect>")

        result = inference.predict(
            self.OWN_INPUT,
            self.PARTNER_INPUT,
            self.DIALOGUE,
        )

        self.assertTrue(result.is_supported)
        self.assertFalse(result.is_acceptable)
        self.assertEqual(result.category, "disconnect")


if __name__ == "__main__":
    unittest.main()