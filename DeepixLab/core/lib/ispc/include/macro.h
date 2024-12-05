#define CAT(a,...) CAT_impl(a, __VA_ARGS__)
#define CAT_impl(a,...) a ## __VA_ARGS__

#define CAT2(a,b) CAT(a,b)
#define CAT3(a,b,c) CAT(a,CAT(b,c))
#define CAT4(a,b,c,d) CAT(a,CAT(b,CAT(c,d)))
#define CAT5(a,b,c,d,e) CAT(a,CAT(b,CAT(c,CAT(d,e))))
