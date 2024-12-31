if(USE_CUDA)
  set(flow_sources
      flows/V/V.cu flows/flows.cc
      PARENT_SCOPE)
else()
  set(flow_sources
      flows/V/V.cc flows/flows.cc
      PARENT_SCOPE)
endif()
