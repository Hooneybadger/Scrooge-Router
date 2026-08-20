// SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
// SPDX-License-Identifier: Apache-2.0

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define FNV_OFFSET UINT64_C(14695981039346656037)
#define FNV_PRIME UINT64_C(1099511628211)

static uint64_t fnv_update(uint64_t digest, const unsigned char *bytes,
                           Py_ssize_t length) {
    Py_ssize_t index;
    for (index = 0; index < length; ++index) {
        digest ^= bytes[index];
        digest *= FNV_PRIME;
    }
    return digest;
}

static PyObject *stable_hash(PyObject *self, PyObject *value) {
    const char *bytes;
    Py_ssize_t length;
    uint64_t digest = FNV_OFFSET;

    (void)self;
    bytes = PyUnicode_AsUTF8AndSize(value, &length);
    if (bytes == NULL) {
        return NULL;
    }
    digest = fnv_update(digest, (const unsigned char *)bytes, length);
    return PyLong_FromUnsignedLongLong(digest);
}

static int utf8_write(Py_UCS4 value, unsigned char *output) {
    if (value <= 0x7f) {
        output[0] = (unsigned char)value;
        return 1;
    }
    if (value <= 0x7ff) {
        output[0] = (unsigned char)(0xc0 | (value >> 6));
        output[1] = (unsigned char)(0x80 | (value & 0x3f));
        return 2;
    }
    if (value <= 0xffff) {
        output[0] = (unsigned char)(0xe0 | (value >> 12));
        output[1] = (unsigned char)(0x80 | ((value >> 6) & 0x3f));
        output[2] = (unsigned char)(0x80 | (value & 0x3f));
        return 3;
    }
    output[0] = (unsigned char)(0xf0 | (value >> 18));
    output[1] = (unsigned char)(0x80 | ((value >> 12) & 0x3f));
    output[2] = (unsigned char)(0x80 | ((value >> 6) & 0x3f));
    output[3] = (unsigned char)(0x80 | (value & 0x3f));
    return 4;
}

enum token_kind {
    TOKEN_SKIP = 0,
    TOKEN_ASCII = 1,
    TOKEN_HANGUL = 2,
    TOKEN_DECIMAL = 3,
    TOKEN_SYMBOL = 4,
};

static enum token_kind classify(Py_UCS4 value) {
    if ((value >= 'A' && value <= 'Z') ||
        (value >= 'a' && value <= 'z')) {
        return TOKEN_ASCII;
    }
    if (value >= 0xac00 && value <= 0xd7a3) {
        return TOKEN_HANGUL;
    }
    if (Py_UNICODE_ISDECIMAL(value)) {
        return TOKEN_DECIMAL;
    }
    if (value != '_' && !Py_UNICODE_ISALNUM(value) &&
        !Py_UNICODE_ISSPACE(value)) {
        return TOKEN_SYMBOL;
    }
    return TOKEN_SKIP;
}

static int make_token(int kind, int unicode_kind, const void *unicode_data,
                      Py_ssize_t start, Py_ssize_t end,
                      unsigned char **token, Py_ssize_t *capacity,
                      Py_ssize_t *token_length) {
    static const unsigned char number[] = "<number>";
    Py_ssize_t index;
    Py_ssize_t required;
    unsigned char *cursor;
    unsigned char *resized;

    required = kind == TOKEN_DECIMAL
                   ? (Py_ssize_t)(sizeof(number) - 1)
                   : 4 * (end - start);
    if (*capacity < required) {
        resized = PyMem_Realloc(*token, (size_t)required);
        if (resized == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        *token = resized;
        *capacity = required;
    }
    if (kind == TOKEN_DECIMAL) {
        *token_length = required;
        memcpy(*token, number, (size_t)*token_length);
        return 0;
    }
    cursor = *token;
    for (index = start; index < end; ++index) {
        Py_UCS4 value = PyUnicode_READ(unicode_kind, unicode_data, index);
        if (kind == TOKEN_ASCII && value >= 'A' && value <= 'Z') {
            value += 'a' - 'A';
        }
        cursor += utf8_write(value, cursor);
    }
    *token_length = cursor - *token;
    return 0;
}

static PyObject *hashed_features(PyObject *self, PyObject *args) {
    static const unsigned char unigram_prefix[] = "w1:";
    static const unsigned char bigram_prefix[] = "w2:";
    PyObject *text;
    PyObject *result = NULL;
    Py_ssize_t bins;
    Py_ssize_t length;
    Py_ssize_t index = 0;
    int unicode_kind;
    void *unicode_data;
    double *values = NULL;
    unsigned char *previous = NULL;
    Py_ssize_t previous_length = 0;
    Py_ssize_t previous_capacity = 0;
    unsigned char *current = NULL;
    Py_ssize_t current_capacity = 0;
    double norm = 0.0;

    (void)self;
    if (!PyArg_ParseTuple(args, "On:hashed_features", &text, &bins)) {
        return NULL;
    }
    if (!PyUnicode_Check(text)) {
        PyErr_SetString(PyExc_TypeError, "text must be str");
        return NULL;
    }
    if (bins <= 0 || (bins & (bins - 1)) != 0) {
        PyErr_SetString(PyExc_ValueError, "bins must be a positive power of two");
        return NULL;
    }
    if (PyUnicode_READY(text) < 0) {
        return NULL;
    }
    length = PyUnicode_GET_LENGTH(text);
    unicode_kind = PyUnicode_KIND(text);
    unicode_data = PyUnicode_DATA(text);
    values = PyMem_Calloc((size_t)bins, sizeof(double));
    if (values == NULL) {
        return PyErr_NoMemory();
    }
    while (index < length) {
        Py_UCS4 value = PyUnicode_READ(unicode_kind, unicode_data, index);
        enum token_kind kind = classify(value);
        Py_ssize_t start = index;
        Py_ssize_t end;
        Py_ssize_t current_length = 0;
        uint64_t digest;

        if (kind == TOKEN_SKIP) {
            ++index;
            continue;
        }
        end = index + 1;
        if (kind != TOKEN_SYMBOL) {
            while (end < length &&
                   classify(PyUnicode_READ(unicode_kind, unicode_data, end)) ==
                       kind) {
                ++end;
            }
        }
        index = end;
        if (make_token(kind, unicode_kind, unicode_data, start, end,
                       &current, &current_capacity, &current_length) < 0) {
            goto cleanup;
        }
        digest = fnv_update(FNV_OFFSET, unigram_prefix,
                            (Py_ssize_t)(sizeof(unigram_prefix) - 1));
        digest = fnv_update(digest, current, current_length);
        values[digest & (uint64_t)(bins - 1)] +=
            (digest & (UINT64_C(1) << 63)) ? -1.0 : 1.0;
        if (previous != NULL) {
            digest = fnv_update(FNV_OFFSET, bigram_prefix,
                                (Py_ssize_t)(sizeof(bigram_prefix) - 1));
            digest = fnv_update(digest, previous, previous_length);
            digest = fnv_update(digest, (const unsigned char *)"\x1f", 1);
            digest = fnv_update(digest, current, current_length);
            values[digest & (uint64_t)(bins - 1)] +=
                (digest & (UINT64_C(1) << 63)) ? -1.0 : 1.0;
        }
        {
            unsigned char *swapped_token = previous;
            Py_ssize_t swapped_capacity = previous_capacity;
            previous = current;
            previous_capacity = current_capacity;
            current = swapped_token;
            current_capacity = swapped_capacity;
        }
        previous_length = current_length;
    }
    for (index = 0; index < bins; ++index) {
        norm += values[index] * values[index];
    }
    norm = sqrt(norm);
    result = PyTuple_New(bins);
    if (result == NULL) {
        goto cleanup;
    }
    for (index = 0; index < bins; ++index) {
        PyObject *item = PyFloat_FromDouble(norm == 0.0 ? values[index]
                                                        : values[index] / norm);
        if (item == NULL) {
            Py_CLEAR(result);
            goto cleanup;
        }
        PyTuple_SET_ITEM(result, index, item);
    }

cleanup:
    PyMem_Free(previous);
    PyMem_Free(current);
    PyMem_Free(values);
    return result;
}

static PyMethodDef methods[] = {
    {"stable_hash", stable_hash, METH_O, "Return the unsigned FNV-1a 64 hash."},
    {"hashed_features", hashed_features, METH_VARARGS,
     "Return exact normalized signed FNV word unigram/bigram bins."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_fnvfast",
    "Exact native FNV-1a helper for the final router.",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__fnvfast(void) {
    return PyModule_Create(&module);
}
