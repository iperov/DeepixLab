extern int _fltused = 0;
extern size_t __chkstk(size_t size) {__asm__(
"lea (%rsp,%rax), %rax \r\n"
"add (0x8), %rax \r\n"
"retn \r\n"
    
"push %rcx \r\n"
"push %rax \r\n"

"lea (%rsp,0x10), %rcx \r\n"

"sub %rcx, %rax \r\n"
"test (%rcx), %rcx \r\n"

"pop %rax \r\n"
"pop %rcx \r\n"
"retn \r\n"

);}

    /*
    __asm
    {
    push    rcx
    push    rax
    cmp     rax, 0x1000
    lea     rcx, [rsp+0x10+arg_0]
    jb      @loc_10003247

@loc_1000322F:                           
    sub     rcx, 0x1000
    test    [rcx], rcx
    sub     rax, 0x1000
    cmp     rax, 0x1000
    ja      @loc_1000322F

@loc_10003247:                           
    sub     rcx, rax
    test    [rcx], rcx
    pop     rax
    pop     rcx
    retn
    }
    */
    
