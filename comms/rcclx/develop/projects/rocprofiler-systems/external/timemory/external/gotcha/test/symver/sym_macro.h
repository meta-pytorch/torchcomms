#ifndef GOTCHA_SYM_MACRO_H
#define GOTCHA_SYM_MACRO_H
#ifdef __has_attribute
#if __has_attribute(__symver__)
#define SYMVER_ATTRIBUTE(sym, symver) __attribute__((__symver__(#symver)))
#endif
#endif
#ifndef SYMVER_ATTRIBUTE
#define SYMVER_ATTRIBUTE(sym, symver)  \
  __asm__("\t.globl  __" #sym          \
          "\n"                         \
          "\t.equiv  __" #sym "," #sym \
          "\n"                         \
          "\t.symver __" #sym "," #symver);
#endif
#endif  // GOTCHA_SYM_MACRO_H
