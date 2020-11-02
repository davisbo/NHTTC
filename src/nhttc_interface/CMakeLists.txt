set( INTERFACE_HEADERS nhttc_interface.h )
set( INTERFACE_SOURCE nhttc_interface.cpp )

add_library(NHTTC_INTERFACE SHARED ${INTERFACE_SOURCE} ${INTERFACE_HEADERS})