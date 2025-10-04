#include "tokenizer.h"

int char_to_token(char c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    return 26; // unknown token
}

char token_to_char(int t) {
    if (t >= 0 && t < 26) return 'a' + t;
    return '?';
}
