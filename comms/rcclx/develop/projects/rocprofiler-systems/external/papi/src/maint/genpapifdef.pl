#!/usr/bin/perl

##
## Copyright (C) by Innovative Computing Laboratory
##     See COPYRIGHT in top level directory
##

use warnings;
use strict;

my $debug = 0;
my $compiler;
my $papi_h = "papi.h";
my $events_h = "papiStdEventDefs.h";
my ($header, $value, $operator, $trailer) = ("[-+() ]*", "[A-Za-z_0-9]+", "[<>|&+-]*", ".*");

&parse_script_args(@ARGV);

my %papi_defs = &parse_papi_defs($papi_h);
my %papi_presets = &parse_papi_presets($events_h);

&write_defs(%papi_defs);
&write_presets(%papi_presets);


# Subroutines
sub parse_script_args {
    my @argv = @_;

    foreach $_ (@argv) {
        if (/-c/) {
            $compiler = "fort";
        } elsif (/-f77/) {
            $compiler = "f77";
        } elsif (/-f90/) {
            $compiler = "f90";
        } elsif (/-debug/) {
            $debug = 1;
        } else {
            die "Unrecognized argument $_\n";
        }
    }
}

sub parse_papi_defs {

    my $filename = $_[0];
    my %papi_defs = ();

    open (my $fh_in, "<$filename") || die "Unable to open $filename\n";

    while (my $line = <$fh_in>) {

        $line =~ s/\/\*(.*)//;

        # handle PAPI_VERSION explicitly
        if ($line =~ /^\s*#\s*define\s+(PAPI_[A-Z_0-9]+)\s+PAPI_VERSION_NUMBER\(([0-9]+),([0-9]+),([0-9])+,([0-9]+)\)/) {
            $papi_defs{'PAPI_VERSION'} = ($2 << 24) | ($3 << 16) | ($4 << 8) | $5;
        }
        # match: define PAPI_XXX (value)
        elsif ($line =~ /^\s*#\s*define\s+(PAPIF?_[A-Z_0-9]+)\s+(.*)/) {
            my ($name, $content, $eval_string) = ($1, $2, "");

            # Search for PAPI_XXX definitions and replace them in eval_string;
            # then evaluate eval_string
            while ($content =~ /($header)\s*($value)\s*($operator)\s*($trailer)/) {
                my ($h, $v, $o, $t) = ($1, $2, $3, $4);

                $eval_string .= $h.(exists($papi_defs{$v}) ?
                    $papi_defs{$v} :
                    ($v =~ /0x/) ? hex($v) : $v).$o;

                $content = $t;
            }

            $eval_string .= $content;
            $papi_defs{$name} = eval $eval_string;
            print STDERR ">> $name = $eval_string\n" if $debug;
        }
        # match: enum NAME {
        elsif ($line =~ /^\s*enum\s*[A-Za-z0-9_]*\s*(.*)/ ||
               $line =~ /^\s*enum\s*[A-Za-z0-9_]*\s*{\s*(.*)/ ||
               $line =~ /^\s*typedef\s*enum\s*[A-Za-z0-9_]*\s*(.*)/ ||
               $line =~ /^\s*typedef\s*enum\s*[A-Za-z0-9_]*\s*{\s*(.*)/) {

            # Eat until we find the closing right brace
            my $enum_line = $1;
            while (! ($enum_line =~ /}/)) {
                my $newline = <$fh_in>;
                $newline =~ s/\r*\n//;
                $enum_line .= $newline;
            }

            my ($name, $content, $prev_key) = ("", "", "");
            my @enum_array = split /,/, $enum_line;

            foreach my $item (@enum_array) {

                my $eval_string = "";

                # clean up white comments, white spaces and braces
                $item =~ s/\/\*.+\*\///s;
                $item =~ s/\s+//;
                $item =~ s/{//g;
                $item =~ s/}.*//;

                if ($item =~ /(PAPI_[A-Z_0-9]+)\s*=(.*)/) {
                    ($name, $content) = ($1, $2);

                    # Search for PAPI_XXX definitions and replace them in eval_string;
                    # then evaluate eval_string
                    while ($content =~ /\s*($header)\s*($value)\s*($operator)\s*($trailer)/) {
                        my ($h, $v, $o, $t) = ($1, $2, $3, $4);

                        $eval_string .= $h.(exists($papi_defs{$v}) ?
                            $papi_defs{$v} :
                            ($v =~ /0x/) ? hex($v) : $v).$o;

                        $content = $t;
                    }
                }
                elsif ($item =~ /(PAPI_[A-Z_0-9]+)/) {
                    ($name, $content) = ($item, "");
                    $eval_string .= (($prev_key eq "") ? 0 : $papi_defs{$prev_key} + 1);
                } else {
                    next;
                }

                $eval_string .= $content;
                $papi_defs{$name} = eval $eval_string;
                print STDERR ">> $name = $eval_string\n" if $debug;
                $prev_key = $name;
            }
        }
    }

    close($fh_in);
    return %papi_defs;
}

sub parse_papi_presets {

    my $filename = $_[0];
    my %papi_presets = ();

    open(my $fh_in, "<$filename") || die "Unable to open $filename\n";

    # FIXME: this implementation is not generic enough
    while (my $line = <$fh_in>) {

        # cleanup comments
        $line =~ s/\/\*(.*)\*\)//;

        # match: enum NAME {
        if ($line =~ /^\s*enum\s*[A-Za-z0-9_]*\s*(.*)/ ||
            $line =~ /^\s*enum\s*[A-Za-z0-9_]*\s*{\s*(.*)/ ||
            $line =~ /^\s*typedef\s*enum\s*[A-Za-z0-9_]*\s*(.*)/ ||
            $line =~ /^\s*typedef\s*enum\s*[A-Za-z0-9_]*\s*{\s*(.*)/) {
            # Eat until we find the closing right brace
            my $enum_line = $1;
            while (! ($enum_line =~ /}/)) {
                my $newline = <$fh_in>;
                $newline =~ s/\r*\n//;
                $enum_line .= $newline;
            }

            my $prev_key = "";

            while (1) {
                # match: PAPI_XXX_idx,
                if ($enum_line =~ /\s*(PAPI_[A-Z_0-9]+)_idx(.*)/) {
                    $papi_presets{$1} = (("$prev_key" eq "") ? -2147483648 : $papi_presets{$prev_key} + 1);
                    $prev_key = $1;
                    $enum_line = $2;
                }
                # match: closing right brace
                elsif ($enum_line =~ /}/) {
                    last;
                }
            }
        }
    }

    close($fh_in);
    return %papi_presets;
}

sub write_defs {
    my %defs = @_;

    if ("$compiler" eq "fort") {
        &write_defs_fort(%defs);
    } elsif ("$compiler" eq "f77") {
        &write_defs_f77(%defs);
    } else {
        &write_defs_f90(%defs);
    }
}

sub write_presets {
    my %presets = @_;

    if ("$compiler" eq "fort") {
        &write_presets_fort(%presets);
    } elsif ("$compiler" eq "f77") {
        &write_presets_f77(%presets);
    } else {
        &write_presets_f90(%presets);
    }
}

sub write_defs_fort {
    my %defs = @_;

    printf STDOUT "C\n";
    printf STDOUT "C This file contains defines required by the PAPI Fortran interface.\n";
    printf STDOUT "C It is automatically generated by genpapifdef.pl.\n";
    printf STDOUT "C DO NOT modify its content and expect the changes to stick.\n";
    printf STDOUT "C Changes MUST be made in genpapifdef.pl instead.\n";
    printf STDOUT "C Content is extracted from define and enum statements in papi.h\n";
    printf STDOUT "C All other content is ignored.\n";
    printf STDOUT "C\n\n";

    printf STDOUT "C\n";
    printf STDOUT "C General purpose defines\n";
    printf STDOUT "C\n\n";

    foreach my $key (keys %defs) {
        # skip unneeded definition
        if ($key =~ /PAPI_MH_/ || $key =~ /PAPI_PRESET_/ || $key =~ /PAPI_DEF_ITIMER/) { next; }
        printf STDOUT "#define %-18s %s\n", $key, ($papi_defs{$key} == 0x80000000) ? "((-2147483647) - 1)" : $papi_defs{$key};
    }
}

sub write_defs_f77 {
    my %defs = @_;

    printf STDOUT "!\n";
    printf STDOUT "! This file contains defines required by the PAPI Fortran interface.\n";
    printf STDOUT "! It is automatically generated by genpapifdef.pl.\n";
    printf STDOUT "! DO NOT modify its content and expect the changes to stick.\n";
    printf STDOUT "! Changes MUST be made in genpapifdef.pl instead.\n";
    printf STDOUT "! Content is extracted from define and enum statements in papi.h\n";
    printf STDOUT "! All other content is ignored.\n";
    printf STDOUT "!\n\n";

    printf STDOUT "!\n";
    printf STDOUT "! General purpose defines\n";
    printf STDOUT "!\n\n";

    foreach my $key (keys %defs) {
        # skip unneeded definition
        if ($key =~ /PAPI_MH_/ || $key =~ /PAPI_PRESET_/ || $key =~ /PAPI_DEF_ITIMER/) { next; }
        printf STDOUT "INTEGER %-18s\nPARAMETER(%s=%s)\n", $key, $key, ($papi_defs{$key} == 0x80000000) ? "((-2147483647) - 1)" : $papi_defs{$key};
    }
}

sub write_defs_f90 {
    my %defs = @_;

    printf STDOUT "!\n";
    printf STDOUT "! This file contains defines required by the PAPI Fortran interface.\n";
    printf STDOUT "! It is automatically generated by genpapifdef.pl.\n";
    printf STDOUT "! DO NOT modify its content and expect the changes to stick.\n";
    printf STDOUT "! Changes MUST be made in genpapifdef.pl instead.\n";
    printf STDOUT "! Content is extracted from define and enum statements in papi.h\n";
    printf STDOUT "! All other content is ignored.\n";
    printf STDOUT "!\n\n";

    printf STDOUT "!\n";
    printf STDOUT "! General purpose defines\n";
    printf STDOUT "!\n\n";

    foreach my $key (keys %defs) {
        # skip unneeded definition
        if ($key =~ /PAPI_MH_/ || $key =~ /PAPI_PRESET_/ || $key =~ /PAPI_DEF_ITIMER/) { next; }
        printf STDOUT "INTEGER, PARAMETER :: %-18s = %s\n", $key, ($papi_defs{$key} == 0x80000000) ? "((-2147483647) - 1)" : $papi_defs{$key};
    }
}

sub write_presets_fort {
    my %presets = @_;

    printf STDOUT "\n";
    printf STDOUT "C\n";
    printf STDOUT "C PAPI preset event values\n";
    printf STDOUT "C\n\n";

    foreach my $key (keys %presets) {
        if ($papi_presets{$key} == -2147483648) {
            printf STDOUT "#define %-18s ((-2147483647) - 1)\n", $key;
        } else {
            printf STDOUT "#define %-18s %s\n", $key, $papi_presets{$key};
        }
    }
}

sub write_presets_f77 {
    my %presets = @_;

    printf STDOUT "\n";
    printf STDOUT "!\n";
    printf STDOUT "! PAPI preset event values\n";
    printf STDOUT "!\n\n";

    foreach my $key (keys %presets) {
        if ($papi_presets{$key} == -2147483648) {
            printf STDOUT "INTEGER %-18s\nPARAMETER(%s=(-2147483647) - 1)\n", $key, $key;
        } else {
            printf STDOUT "INTEGER %-18s\nPARAMETER(%s=%s)\n", $key, $key, $papi_presets{$key};
        }
    }
}

sub write_presets_f90 {
    my %presets = @_;

    printf STDOUT "\n";
    printf STDOUT "!\n";
    printf STDOUT "! PAPI preset event values\n";
    printf STDOUT "!\n\n";

    foreach my $key (keys %presets) {
        if ($papi_presets{$key} == -2147483648) {
            printf STDOUT "INTEGER, PARAMETER :: %-18s = ((-2147483647) - 1)\n", $key;
        } else {
            printf STDOUT "INTEGER, PARAMETER :: %-18s = %s\n", $key, $papi_presets{$key};
        }
    }
}
