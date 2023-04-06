#!/usr/bin/env python
# coding: utf8

""" TO DOCUMENT """

from pytest import raises

from spleeter.model.provider import ModelProvider


def test_checksum():
    """Test archive checksum index retrieval."""
    provider = ModelProvider.default()
    assert (
        provider.checksum("2stems")
        == "f3a90b39dd2874269e8b05a48a86745df897b848c61f3958efc80a39152bd692"
    )
    assert (
        provider.checksum("4stems")
        == "3adb4a50ad4eb18c7c4d65fcf4cf2367a07d48408a5eb7d03cd20067429dfaa8"
    )
    assert (
        provider.checksum("5stems")
        == "25a1e87eb5f75cc72a4d2d5467a0a50ac75f05611f877c278793742513cc7218"
    )
    with raises(ValueError):
        provider.checksum("laisse moi stems stems stems")
