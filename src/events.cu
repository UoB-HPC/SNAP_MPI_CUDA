
#include "events.h"


void load_event(cudaEvent_t e)
{
    cudaEventRecord(e);
    check_cuda("Record event");
}

