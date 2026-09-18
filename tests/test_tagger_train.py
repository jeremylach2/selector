from selector.tagger.train import _format_example, _read_jsonl


class _FakeTokenizer:
    """Mimics the slice of the HF tokenizer interface _format_example uses,
    so this test doesn't need the real (heavy) transformers dependency."""

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        # One token per word - deterministic and easy to assert on.
        return {"input_ids": list(range(len(text.split())))}


def test_format_example_masks_prompt_tokens():
    tokenizer = _FakeTokenizer()
    example = {"prompt": "a b c", "completion": "d e"}

    formatted = _format_example(tokenizer, example, max_length=100)

    n_prompt_tokens = len(["a", "b", "c"])
    assert formatted["labels"][:n_prompt_tokens] == [-100] * n_prompt_tokens
    assert formatted["labels"][n_prompt_tokens:] != [-100] * (len(formatted["labels"]) - n_prompt_tokens)
    assert len(formatted["input_ids"]) == len(formatted["labels"]) == len(formatted["attention_mask"])


def test_format_example_truncates_to_max_length():
    tokenizer = _FakeTokenizer()
    example = {"prompt": "one two three four five", "completion": "six seven eight"}

    formatted = _format_example(tokenizer, example, max_length=4)

    assert len(formatted["input_ids"]) == 4
    assert len(formatted["labels"]) == 4


def test_read_jsonl_parses_each_line(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")

    rows = _read_jsonl(path)

    assert rows == [{"a": 1}, {"a": 2}]
