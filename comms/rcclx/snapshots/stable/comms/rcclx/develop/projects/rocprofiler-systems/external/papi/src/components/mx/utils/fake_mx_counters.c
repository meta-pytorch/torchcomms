#include <stdio.h>

/* This program fakes the output of the mx_counters util */
/* for testing purposes */

#define MAX_LABELS 101

char labels[MAX_LABELS][100]={
"            Lanai uptime (seconds)",
"         Counters uptime (seconds)",
"                 Bad CRC8 (Port 0)",
"                Bad CRC32 (Port 0)",
"         Unstripped route (Port 0)",
"         pkt_desc_invalid (Port 0)",
"          recv_pkt_errors (Port 0)",
"            pkt_misrouted (Port 0)",
"                  data_src_unknown",
"                    data_bad_endpt",
"                 data_endpt_closed",
"                  data_bad_session",
"                   push_bad_window",
"                    push_duplicate",
"                     push_obsolete",
"                  push_race_driver",
"        push_bad_send_handle_magic",
"                push_bad_src_magic",
"                     pull_obsolete",
"              pull_notify_obsolete",
"                  pull_race_driver",
"                  pull_notify_race",
"                      ack_bad_type",
"                     ack_bad_magic",
"                   ack_resend_race",
"                          Late ack",
"           ack_nack_frames_in_pipe",
"                    nack_bad_endpt",
"                 nack_endpt_closed",
"                  nack_bad_session",
"                  nack_bad_rdmawin",
"                  nack_eventq_full",
"                  send_bad_rdmawin",
"                   connect_timeout",
"               connect_src_unknown",
"                   query_bad_magic",
"                   query_timed_out",
"                 query_src_unknown",
"                Raw sends (Port 0)",
"             Raw receives (Port 0)",
"    Raw oversized packets (Port 0)",
"                  raw_recv_overrun",
"                      raw_disabled",
"                      connect_send",
"                      connect_recv",
"                 ack_send (Port 0)",
"                 ack_recv (Port 0)",
"                push_send (Port 0)",
"                push_recv (Port 0)",
"               query_send (Port 0)",
"               query_recv (Port 0)",
"               reply_send (Port 0)",
"               reply_recv (Port 0)",
"            query_unknown (Port 0)",
"            query_unknown (Port 0)",
"           data_send_null (Port 0)",
"          data_send_small (Port 0)",
"         data_send_medium (Port 0)",
"           data_send_rndv (Port 0)",
"           data_send_pull (Port 0)",
"           data_recv_null (Port 0)",
"   data_recv_small_inline (Port 0)",
"     data_recv_small_copy (Port 0)",
"         data_recv_medium (Port 0)",
"           data_recv_rndv (Port 0)",
"           data_recv_pull (Port 0)",
"   ether_send_unicast_cnt (Port 0)",
" ether_send_multicast_cnt (Port 0)",
"     ether_recv_small_cnt (Port 0)",
"       ether_recv_big_cnt (Port 0)",
"                     ether_overrun",
"                   ether_oversized",
"              data_recv_no_credits",
"                    Packets resent",
"  Packets dropped (data send side)",
"              Mapper routes update",
"         Route dispersion (Port 0)",
"               out_of_send_handles",
"               out_of_pull_handles",
"               out_of_push_handles",
"                  medium_cont_race",
"                  cmd_type_unknown",
"                 ureq_type_unknown",
"                Interrupts overrun",
"         Waiting for interrupt DMA",
"         Waiting for interrupt Ack",
"       Waiting for interrupt Timer",
"                   Slabs recycling",
"                    Slabs pressure",
"                  Slabs starvation",
"               out_of_rdma handles",
"                       eventq_full",
"              buffer_drop (Port 0)",
"              memory_drop (Port 0)",
"    Hardware flow control (Port 0)",
"(Devel) Simulated packets lost (Port 0)",
"   (Logging) Logging frames dumped",
"                   Wake interrupts",
"               Averted wakeup race",
"                 Dma metadata race",
"                               foo",
};

int main(int argc, char **argv) {
  
  int i,multiplier=1;

  FILE *fff;

  fff=fopen("state","r");
  if (fff!=NULL) {
    fscanf(fff,"%d",&multiplier);
    fclose(fff);
  }

  fff=fopen("state","w");
  if (fff!=NULL) {
    fprintf(fff,"%d\n",multiplier+1);
    fclose(fff);
  }

  printf("1 ports\n");
  for(i=0;i<MAX_LABELS;i++) {
    printf("%s:%12d (%#x)\n",labels[i],i*multiplier,i*multiplier);
  }
  return 0;
}

