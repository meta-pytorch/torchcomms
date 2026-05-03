/**
 * PAPI - Indent profile. <p>
 *
 * The purpose of this file is to standardize the PAPI's source code style.
 * Every new/modified source should be formatted with indent using this
 * profile before it is checked in again.
 * 
 * @name       .indent.pro
 * 
 * @version    $Revision$ <br>
 *             $Date$ <br>
 *             $Author$
 * 
 * @author     Heike Jagode
 */




/* use tabs */
--use-tabs

/* set tab size to 4 spaces */
--tab-size4

/* set indentation level to 4 spaces, and these will be turned into
 * tabs by default */
--indent-level4

/* don't put variables in column 16 */
//--declaration-indentation16




/* maximum length of a line is 80 */
--line-length80

/* breakup the procedure type */
--procnames-start-lines
// --dont-break-procedure-type

/* break long lines after the boolean operators && and || */
--break-after-boolean-operator

/* if long lines are already broken up, GNU indent won't touch them */
--honour-newlines

/* If a line has a left parenthesis which is not closed on that line,
 * then continuation lines will be lined up to start at the character
 * position just after the left parenthesis */
--continue-at-parentheses

/* NO! (see --continue-at-parentheses) */
--continuation-indentation0




/* put braces on line with if, etc.*/
--braces-on-if-line
//--braces-after-if-line

/* put braces on the line after struct declaration lines */
--braces-after-struct-decl-line

/* put braces on the line after function definition lines */
--braces-after-func-def-line

/* indent braces 0 spaces */
--brace-indent0

/* NO extra struct/union brace indentation */
--struct-brace-indentation0

/* NO extra case brace indentation! */
--case-brace-indentation0

/* put a space after and before every parenthesis */
--space-after-parentheses

/* NO extra parentheses indentation in broken lines */
--paren-indentation0




/* blank line causes problems with multi parameter function prototypes */
--no-blank-lines-after-declarations

/* forces blank line after every procedure body */
--blank-lines-after-procedures

/* NO newline is forced after each comma in a declaration */
--no-blank-lines-after-commas

/* allow optional blank lines */
--leave-optional-blank-lines
// --swallow-optional-blank-lines




/* do not put comment delimiters on blank lines */
--no-comment-delimiters-on-blank-lines

/* the maximum comment column is 79 */
--comment-line-length79

/* do not touch comments starting at column 0 */
--dont-format-first-column-comments

/* no extra line comment indentation */
--line-comments-indentation0

/* dont star comments */
--dont-star-comments
// --start-left-side-of-comments

/* comments to the right of the code start at column 30 */
--comment-indentation30

/* comments after declarations start at column 40 */
--declaration-comment-column40

/* comments after #else #endif start at column 8 */
--else-endif-column8




/* Do not cuddle } and the while of a do {} while; */
--dont-cuddle-do-while

/* Do cuddle } and else */
--cuddle-else
//--dont-cuddle-else

/* a case label indentation of 0 */
--case-indentation0

/* put no space after a cast operator */
//--no-space-after-casts

/* no space after function call names;
 * but space after keywords for, it, while */
--no-space-after-function-call-names
//--no-space-after-for
//--no-space-after-if
//--no-space-after-while

/* Do not force space between special statements and semicolon */
--dont-space-special-semicolon
// --space-special-semicolon

/* put a space between sizeof and its argument :TODO: check */
--blank-before-sizeof

/* enable verbose mode */
--verbose
// --no-verbosity




/* NO space between # and preprocessor directives */
// --leave-preprocessor-space

/* format some comments but not all */
// --dont-format-comments

/* NO gnu style as default */
// --gun_style

/* K&R default style */
--k-and-r-style

/* NO Berkeley default style */
// --original

/* read this profile :-) */
// --ignore-profile

