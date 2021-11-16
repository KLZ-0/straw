#include <Python.h>
#include <math.h>

// Actual functions

static PyObject* lpc_residual(PyObject *self, PyObject *args) {
    double p;

    /* This parses the Python argument into a double */
    if(!PyArg_ParseTuple(args, "d", &p)) {
        return NULL;
    }

    /* THE ACTUAL LOGIT FUNCTION */
    p = p/(1-p);
    p = log(p);

    /*This builds the answer back into a python object */
    return Py_BuildValue("d", p);
}

// Module initialization

static PyMethodDef LpcMethods[] = {
    {"compute_residual",
        lpc_residual,
        METH_VARARGS, "compute LPC residual"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "lpc",
    NULL,
    -1,
    LpcMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_lpc(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
