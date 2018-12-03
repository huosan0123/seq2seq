#include <Python.h>
#include <numpy/arrayobject.h>
#include <map>
#include <string>
#include <iostream>
#include <math.h>
using namespace std;

map<string, float> aa_mass = {{"_GO", 0.0},
                              {"_EOS", 0.0},
                              {"A", 71.03711},
                              {"R", 156.10111},
                              {"N", 114.04293},
                              {"Nmod", 115.02695},
                              {"D", 115.02694},
                              {"Cmod", 160.03065},
                              {"E", 129.04259},
                              {"Q", 128.05858},
                              {"Qmod", 129.0426},
                              {"G", 57.02146},
                              {"H", 137.05891},
                              {"I", 113.08406},
                              {"L", 113.08406},
                              {"K", 128.09496},
                              {"M", 131.04049},
                              {"Mmod", 147.0354},
                              {"F", 147.06841},
                              {"P", 97.05276},
                              {"S", 87.03203},
                              {"T", 101.04768},
                              {"W", 186.07931},
                              {"Y", 163.06333},
                              {"V", 99.06841}
                              };

static PyObject *process_seq(PyObject *self, PyObject *args)
{
    PyObject *batch_peps;
    int max_len, batch, EOS;

    if (!PyArg_ParseTuple(args, "Oiii", &batch_peps, &batch, &max_len, &EOS))
        return NULL;
	int nd = 2;
	npy_intp dims[] = {batch, max_len};
    PyArrayObject *batch_seqs = (PyArrayObject *)PyArray_SimpleNew(nd, dims, NPY_INT);
    if (batch_seqs == NULL)
        return NULL;
	
	//init the array with eos
    PyArrayIterObject *iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)batch_seqs);
    while (iter->index < iter->size){
        *(int *)iter->dataptr = EOS;
        PyArray_ITER_NEXT(iter);
    }

    int dest[2];
    for (int i = 0; i < batch; i++){
        PyObject *pep = PyList_GetItem(batch_peps, i);
        int k = PyList_Size(pep);
        for (int j = 0; j < k; j++){
            PyObject *aa = PyList_GetItem(pep, j);
			int id = PyLong_AsLong(aa);

            dest[0] = i;
            dest[1] = j;
            PyArray_ITER_GOTO(iter, dest);
            *(int *)(iter->dataptr) = id;
        }
    }

    Py_DECREF(iter);
    //Py_INCREF(batch_seqs);
    return PyArray_Return(batch_seqs);
}

static PyObject *process_peak(PyObject *self, PyObject *args){
    PyObject *mz_lists, *inten_lists;
    int max_len, batch, res;

    if (!PyArg_ParseTuple(args, "OOiii", &mz_lists, &inten_lists, &batch, &max_len, &res))
        return NULL;
	int nd = 3;
    //int Res = 3000 * res
    int Res = 3000 * res;
	npy_intp dims[] = { batch, max_len, Res };
    PyArrayObject *spectra = (PyArrayObject *)PyArray_ZEROS(nd, dims, NPY_DOUBLE, 0);
    if (spectra == NULL)
        return NULL;

	int i, j;
    PyArrayIterObject *iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)spectra);
    int dest[3];

    for (i = 0; i < batch; i++){
        PyObject *mzs = PyList_GetItem(mz_lists, i);
        PyObject *intens = PyList_GetItem(inten_lists, i);
        int k = PyList_Size(mzs);
        for (j = 0; j < k; j++){
            PyObject *mz = PyList_GetItem(mzs, j);
            double key = PyFloat_AsDouble(mz);
            int id = (int)(key * res);
            PyObject *inten = PyList_GetItem(intens, j);
            double value = PyFloat_AsDouble(inten);

            dest[0] = i;
            dest[1] = j;
            dest[2] = id;
            PyArray_ITER_GOTO(iter, dest);
            *(double *)(iter->dataptr) = value;
        }
    }
    Py_DECREF(iter);
    //Py_INCREF(spectra);
    return PyArray_Return(spectra);
}

static PyMethodDef SpamMethods[] = {
    {"process_seq",  process_seq, METH_VARARGS,
    "convert python list to ndarray."},
    {"process_peak", process_peak, METH_VARARGS, 
    "convert peak list to ndarray vector"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "prepare",   /* name of module */
    "prepare data by C++", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
            or -1 if the module keeps state in global variables. */
    SpamMethods
};

PyMODINIT_FUNC PyInit_prepare(void)
{
    import_array();
    return PyModule_Create(&spammodule);
}
