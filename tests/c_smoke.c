#include "blitz/blitz.h"
#include <stdio.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"C ABI failure line %d\n",__LINE__);return 1;}}while(0)
static int cancel(void* p){return *(int*)p;}
int main(void) {
    float vertices[]={0,0,0,1,0,0,0,1,0};uint32_t indices[]={0,1,2};int stop=1;
    struct color_slot {uint8_t prefix;blitz_color_rgba8 color;uint8_t padding;} colors[]={
        {7,{0,128,255,13},8},{9,{255,0,17,255},10},{11,{24,31,128,0},12}};
    CHECK(sizeof(blitz_color_rgba8)==4&&blitz_abi_version()==3);
    blitz_mesh m={0};m.struct_size=sizeof(m);m.abi_version=BLITZ_ABI_VERSION;m.positions=(blitz_stream){vertices,3,12};m.indices=indices;m.index_count=3;
    m.colors=(blitz_stream){&colors[0].color,3,sizeof(colors[0])};
    blitz_settings s;CHECK(blitz_settings_init(&s,sizeof(s))==BLITZ_OK);s.levels=2;s.base_pixels=32;s.last_pixels=8;s.cancelled=cancel;s.user_data=&stop;
    blitz_result* r=NULL;char error[128];CHECK(blitz_generate(&m,&s,&r,error,sizeof(error))==BLITZ_CANCELLED);
    CHECK(r&&blitz_result_lod_count(r)==2);blitz_lod_info l={0};l.struct_size=sizeof(l);
    CHECK(blitz_result_runtime_lod_count(r)==1&&blitz_result_runtime_lod_index(r,0)==0);
    CHECK(blitz_result_runtime_lod_index(r,1)==SIZE_MAX&&blitz_result_runtime_lod_index(NULL,0)==SIZE_MAX);
    CHECK(blitz_result_runtime_lod_count(NULL)==0);
    CHECK(blitz_result_lod(r,1,&l)==BLITZ_OK);CHECK(l.mesh.positions.data==vertices&&l.shared_vertices&&l.passed);
    CHECK(l.mesh.colors.data==&colors[0].color&&l.mesh.colors.stride==sizeof(colors[0]));
    CHECK(l.reference_triangles==1);
    blitz_storage_info storage={0};storage.struct_size=sizeof(storage);
    CHECK(blitz_result_storage(r,&storage)==BLITZ_OK);
    CHECK(storage.source_vertex_bytes==48&&storage.added_vertex_bytes==0&&storage.index_bytes==12&&storage.total_bytes==60);
    CHECK(blitz_result_storage(NULL,&storage)==BLITZ_INVALID_ARGUMENT);
    CHECK(colors[0].color.g==128&&colors[1].color.a==255&&colors[2].color.a==0&&colors[2].padding==12);
    CHECK(blitz_result_lod(r,2,&l)==BLITZ_INVALID_ARGUMENT);blitz_result_destroy(r);blitz_result_destroy(NULL);
    m.abi_version=2;CHECK(blitz_generate(&m,&s,&r,error,sizeof(error))==BLITZ_INVALID_ARGUMENT&&r==NULL);m.abi_version=BLITZ_ABI_VERSION;
    m.colors.stride=3;CHECK(blitz_generate(&m,&s,&r,error,sizeof(error))==BLITZ_INVALID_ARGUMENT&&r==NULL);m.colors.stride=sizeof(colors[0]);
    s.abi_version=42;CHECK(blitz_generate(&m,&s,&r,error,sizeof(error))==BLITZ_INVALID_ARGUMENT&&r==NULL);
    CHECK(blitz_generate(NULL,NULL,&r,error,sizeof(error))==BLITZ_INVALID_ARGUMENT);return 0;
}
