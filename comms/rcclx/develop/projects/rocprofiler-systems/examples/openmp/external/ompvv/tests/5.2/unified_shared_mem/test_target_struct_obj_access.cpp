//===-- test_target_struct_obj_access.cpp ------------------------------===//
//
// OpenMP API Version 5.2
//
// Description
// This test case tests the working of the following aspects:
// 1) Working of "requires unified_shared_memory"
// 2) Testing different ways to access struct objects on gpu
//===-------------------------------------------------------------------===//


#include <iostream>
#include <string>
#include <omp.h>
#include "ompvv.h"

#pragma omp requires unified_shared_memory
#pragma omp begin declare target
struct Employee1 {
  std::string name;
  int Id;
  int Age;
};


struct Employee2 {
  std::string name;
  int Id;
  int Age;
} Emp;

struct Employee3 {
  std::string name = "Murthy";
  int Id = 2233;
  int Age = 27;
};
#pragma omp end declare target

int main() {
  OMPVV_TEST_OFFLOADING;
  int Errors = 0, errors = 0;
  // Need to check the size of structure
  Employee3 EmpPreInit;
  int Sz = sizeof(EmpPreInit);
  int defaultDevice = omp_get_default_device();
#pragma omp target map(tofrom: Errors, Sz)
  {
    if (Sz != sizeof(EmpPreInit)) {
      Errors++;
    }
  }
  // The following are the reference variables from host side
  std::string RefStr = "Vikas";
  int RefId = 1234;
  int RefAge = 24;
  int StrSz = RefStr.length();
  Employee1 Em1;
  Em1.name = "Vikas";
  Em1.Id = 1234;
  Em1.Age = 24;

#pragma omp target device(defaultDevice) map(tofrom: Errors)\
        map(to: RefStr, RefId, RefAge, StrSz)
  {
    for (int i = 0; i <= StrSz; ++i) {
      if (Em1.name[i] != RefStr[i]) {
         Errors++;
      }
    }
    if ((RefId != Em1.Id) || (RefAge != Em1.Age)) {
      Errors++;
    }
  }

// The following Emp variable declared along with the structure
  Emp.name = "Manoj";
  Emp.Id = 1564;
  Emp.Age = 34;
  // Updating the reference variables
  RefStr = "Manoj";
  RefId = 1564;
  RefAge = 34;
  StrSz = RefStr.length();
#pragma omp target device(defaultDevice) map(tofrom: Errors)\
        map(to: RefStr, RefId, RefAge, StrSz, Emp)
  {
    for (int i = 0; i <= StrSz; ++i) {
      if (Emp.name[i] != RefStr[i]) {
         Errors++;
      }
    }
    if ((RefId != Emp.Id) || (RefAge != Emp.Age)) {
      Errors++;
    }
  }

  Employee3 Emp3;

  // Updating the reference variables
  RefStr = "Murthy";
  RefId = 2233;
  RefAge = 27;
  StrSz = RefStr.length();

#pragma omp target device(defaultDevice) map(tofrom: Errors)\
        map(to: RefStr, RefId, RefAge, StrSz)
  {
    for (int i = 0; i <= StrSz; ++i) {
      if (Emp3.name[i] != RefStr[i]) {
         Errors++;
      }
    }
    if ((RefId != Emp3.Id) || (RefAge != Emp3.Age)) {
      Errors++;
    }
  }

  // Access struct var initialized at the time of declaration on gpu
  Employee1 EmpInit = {"Kapil", 2143, 37};
  RefStr = "Kapil";
  RefId = 2143;
  RefAge = 37;
  StrSz = RefStr.length();

#pragma omp target device(defaultDevice) map(tofrom: Errors)\
        map(to: RefStr, RefId, RefAge, StrSz)
  {
    for (int i = 0; i <= StrSz; ++i) {
      if (EmpInit.name[i] != RefStr[i]) {
         Errors++;
      }
    }
    if ((RefId != EmpInit.Id) || (RefAge != EmpInit.Age)) {
      Errors++;
    }
  }
  // Access array of structure on gpu
  Employee1 StrArr[3];
  StrArr[0] = {"Kamal", 1122, 21};
  RefStr = "Kamal";
  RefId = 1122;
  RefAge = 21;
  StrSz = RefStr.length();

#pragma omp target device(defaultDevice) map(tofrom: Errors)\
        map(to: RefStr, RefId, RefAge, StrSz)
  {
    for (int i = 0; i <= StrSz; ++i) {
      if (StrArr[0].name[i] != RefStr[i]) {
         Errors++;
      }
    }
    if ((RefId != StrArr[0].Id) || (RefAge != StrArr[0].Age)) {
      Errors++;
    }
  }

  StrArr[1] = {"Kyra", 1123, 22};
  RefStr = "Kyra";
  RefId = 1123;
  RefAge = 22;
  StrSz = RefStr.length();

#pragma omp target device(defaultDevice) map(tofrom: Errors)\
        map(to: RefStr, RefId, RefAge, StrSz)
  {
    for (int i = 0; i <= StrSz; ++i) {
      if (StrArr[1].name[i] != RefStr[i]) {
         Errors++;
      }
    }
    if ((RefId != StrArr[1].Id) || (RefAge != StrArr[1].Age)) {
      Errors++;
    }
  }
  StrArr[2] = {"Myna", 1124, 23};
  RefStr = "Myna";
  RefId = 1124;
  RefAge = 23;
  StrSz = RefStr.length();

#pragma omp target device(defaultDevice) map(tofrom: Errors)\
        map(to: RefStr, RefId, RefAge, StrSz)
  {
    for (int i = 0; i <= StrSz; ++i) {
      if (StrArr[2].name[i] != RefStr[i]) {
         Errors++;
      }
    }
    if ((RefId != StrArr[2].Id) || (RefAge != StrArr[2].Age)) {
      Errors++;
    }
  }

  // Accessing structure using pointer
  Employee1 *StrPtr = &StrArr[0];
  RefStr = "Kamal";
  RefId = 1122;
  RefAge = 21;

#pragma omp target device(defaultDevice) map(tofrom: Errors, StrPtr)\
        map(to: RefStr, RefId, RefAge, StrSz)
  {
    for (int i = 0; i <= StrSz; ++i) {
      if ((*StrPtr).name[i] != RefStr[i]) {
         Errors++;
      }
    }
    if ((RefId != (*StrPtr).Id) || (RefAge != (*StrPtr).Age)) {
      Errors++;
    }
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, (Errors > 0));
  OMPVV_REPORT_AND_RETURN(errors);
}
