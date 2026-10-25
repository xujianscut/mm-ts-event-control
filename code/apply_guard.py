p = "/home/jianxu/nbp_hd/flowNP/gp_hd.py"
s = open(p).read()
old = "    plot(args, model, eval_batches, step if step is not None else '')\n"
new = ("    if getattr(args, 'dim_x', 1) == 1:\n"
       "        plot(args, model, eval_batches, step if step is not None else '')\n")
n = s.count(old)
assert n == 1, ("count=", n)
s = s.replace(old, new)
open(p, "w").write(s)
print("OK edited, occurrences replaced:", n)
