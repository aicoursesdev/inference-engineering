#!/bin/sh
# Make two speech clips with macOS's built-in voice (16 kHz, as Whisper wants).
say -o short.wav --data-format=LEI16@16000 "Your jacket has shipped."
say -o speech.wav --data-format=LEI16@16000 "Hello, this is Tidewater Goods support. Your green rain jacket has shipped from Kochi and should reach you by Friday. If it does not fit, you can return it within thirty days."
