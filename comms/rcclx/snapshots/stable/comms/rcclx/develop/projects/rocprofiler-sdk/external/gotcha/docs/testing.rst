=============
Testing Guide
=============

We can never have enough testing. Any additional tests you can write are always
greatly appreciated.

----------
Unit Tests
----------

Testing new core features within GOTCHA should be implemented in the ``test/unit/gotcha_unit_tests.c`` using the check framework as defined in `<https://libcheck.github.io/check>`_.


Create a new test
^^^^^^^^^^^^^^^^^
We can create a new test using ``START_TEST`` and ``END_TEST`` macros.

.. code-block:: c
    
    START_TEST(sample_test){
    }
    END_TEST

Create a new suite
^^^^^^^^^^^^^^^^^^
These new tests can be added to new suite with code similar to the following.
To add to existing suite, we need use ``tcase_add_test`` api to add the test function to the suite.

.. code-block:: c

    Suite* gotcha_sample_suite(){
      Suite* s = suite_create("Sample");
      TCase* sample_case = configured_case_create("Basic tests");
      tcase_add_test(sample_case, sample_test);
      suite_add_tcase(s, sample_case);
      return s;
    }

Adding suite to runner
^^^^^^^^^^^^^^^^^^^^^^

Within the main function of the ``test/unit/gotcha_unit_tests.c``, the gotcha_sample_suite can be added as follows.

.. code-block:: c

    Suite* sample_suite = gotcha_sample_suite();
    SRunner* sample_runner = srunner_create(sample_suite);
    srunner_run_all(sample_suite, CK_NORMAL);
    num_fails += srunner_ntests_failed(sample_suite);


-------------------------------------
Testing tool specific usage of GOTCHA
-------------------------------------

We should use custom test cases where we are testing the API for GOTCHA for specific features such as filtering libraries, main function bindings, etc.
These test cases can be stored within the ``test`` folder. Look at existing examples such as ``test/stack`` and ``test/dlopen`` to understand how we can implement these tests.

Once you add a self containing test case within ``test``, we can add it to the ``test/CMakeLists.txt``.
