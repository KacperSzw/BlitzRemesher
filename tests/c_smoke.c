#include "blitz/blitz.h"
#include <stdio.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"C ABI failure line %d\n",__LINE__);return 1;}}while(0)
static int cancel(void* p){return *(int*)p;}
int main(void) {
    float vertices[]={0,0,0,1,0,0,0,1,0};uint32_t indices[]={0,1,2};int stop=1;
    blitz_mesh m={0};m.struct_size=sizeof(m);m.abi_version=BLITZ_ABI_VERSION;m.positions=(blitz_stream){vertices,3,12};m.indices=indices;m.index_count=3;
    blitz_settings s;CHECK(blitz_settings_init(&s,sizeof(s))==BLITZ_OK);s.levels=2;s.base_pixels=32;s.last_pixels=8;s.cancelled=cancel;s.user_data=&stop;
    blitz_result* r=NULL;char error[128];CHECK(blitz_generate(&m,&s,&r,error,sizeof(error))==BLITZ_CANCELLED);
    CHECK(r&&blitz_result_lod_count(r)==2);blitz_lod_info l={0};l.struct_size=sizeof(l);
    CHECK(blitz_result_runtime_lod_count(r)==1&&blitz_result_runtime_lod_index(r,0)==0);
    CHECK(blitz_result_runtime_lod_index(r,1)==SIZE_MAX&&blitz_result_runtime_lod_index(NULL,0)==SIZE_MAX);
    CHECK(blitz_result_runtime_lod_count(NULL)==0);
    CHECK(blitz_result_lod(r,1,&l)==BLITZ_OK);CHECK(l.mesh.positions.data==vertices&&l.shared_vertices&&l.passed);
    CHECK(blitz_result_lod(r,2,&l)==BLITZ_INVALID_ARGUMENT);blitz_result_destroy(r);blitz_result_destroy(NULL);
    s.abi_version=42;CHECK(blitz_generate(&m,&s,&r,error,sizeof(error))==BLITZ_INVALID_ARGUMENT&&r==NULL);
    CHECK(blitz_generate(NULL,NULL,&r,error,sizeof(error))==BLITZ_INVALID_ARGUMENT);return 0;
}
