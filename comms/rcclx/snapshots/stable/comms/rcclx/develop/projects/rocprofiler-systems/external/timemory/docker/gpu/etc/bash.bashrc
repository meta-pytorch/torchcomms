# System-wide .bashrc file for interactive bash(1) shells.

# To enable the settings / commands in this file for login shells as well,
# this file has to be sourced in /etc/profile.
PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:/bin"
export PATH

for j in profile bashrc
do
    if [ -d /etc/${j}.d ]; then
        for i in /etc/${j}.d/*.sh; do
            if [ -r ${i} ]; then
                . ${i}
            fi
        done
    fi
done

umask 0000

export MPLBACKEND=agg
export HISTIGNORE='&:bg:fg:ll:ls'
export HISTCONTROL='ignoreboth:erasedups'
export HISTFILESIZE=5000000

if [ -f "/root/.scl.env" ]; then
    . /root/.scl.env
fi


################################################################################
#
#   If not running interactively, don't do anything
#
################################################################################

[ -z "$PS1" ] && return

################################################################################
#
#   Below this point, only available when interactive
#
################################################################################

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, overwrite the one in /etc/profile)
PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '

# Commented out, don't overwrite xterm -T "title" -n "icontitle" by default.
# If this is an xterm set the title to user@host:dir
case "$TERM" in
    xterm*|rxvt*)
      PROMPT_COMMAND='echo -ne "\033]0;${USER}@${HOSTNAME}: ${PWD}\007"'
    ;;
    *)
    ;;
esac

# enable bash completion in interactive shells
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

# if the command-not-found package is installed, use it
if [ -x /usr/lib/command-not-found -o -x /usr/share/command-not-found/command-not-found ]; then
    function command_not_found_handle {
            # check because c-n-f could've been removed in the meantime
                if [ -x /usr/lib/command-not-found ]; then
           /usr/lib/command-not-found -- "$1"
                   return $?
                elif [ -x /usr/share/command-not-found/command-not-found ]; then
           /usr/share/command-not-found/command-not-found -- "$1"
                   return $?
        else
           printf "%s: command not found\n" "$1" >&2
           return 127
        fi
    }
fi

export LS_OPTIONS='-CF'

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    export LS_OPTIONS="--color=auto ${LS_OPTIONS}"
    alias dir='dir --color=auto'
    alias vdir='vdir --color=auto'

    alias grep='/bin/grep --color=auto'
    alias rgrep='/bin/rgrep --color=auto'
    alias egrep='/bin/egrep --color=auto'
fi

# colored GCC warnings and errors
export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# environment
export SHELL=$(which bash)
export PS1='[ \[\e[1;33m\]\# \[\e[0;22m\]- \[\e[1;31m\]\u@\h \[\e[0;22m\]- \[\e[1;33m\]\@ \[\e[0;22m\]- \[\e[1;35m\]$(/etc/compute-dir-size.py)\[\e[0;22m\] : \[\e[1;36m\]$(pwd) \[\e[0;22m\]] $ \[\e[m\]\[\e[0;22m\]'

# aliases
alias ls='/bin/ls $LS_OPTIONS'
alias ll='/bin/ls $LS_OPTIONS -l'
alias la='/bin/ls $LS_OPTIONS -la'
alias  l='/bin/ls $LS_OPTIONS -lA'
