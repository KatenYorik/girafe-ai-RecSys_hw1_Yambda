import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]; sub = sys.argv[2]
TEST0 = 280 * 86400          # блок 02: test = сутки 280..300
users = pl.read_parquet(os.path.join(sub, "users.parquet"))["uid"]
cat = set(pl.read_parquet(os.path.join(sub, "catalog.parquet"))["item_id"].to_list())
U = users.implode()

# лайки до теста: первое событие лайка на паре + органичность этого лайка
lk = (pl.scan_parquet(os.path.join(raw, "likes.parquet"))
        .filter((pl.col("timestamp") < TEST0) & pl.col("uid").is_in(U))
        .select("uid", "item_id", "timestamp", "is_organic")
        .collect(engine="streaming")
        .sort("timestamp").group_by(["uid", "item_id"])
        .agg(pl.col("timestamp").first().alias("t_like"),
             pl.col("is_organic").first().alias("org_like")))
print("пар с лайком до теста (3 000 пользователей): %d" % lk.height)

# прослушивания тех же пользователей: первое прослушивание до теста и первое вообще
ls = (pl.scan_parquet(os.path.join(raw, "listens.parquet"))
        .filter(pl.col("uid").is_in(U))
        .select("uid", "item_id", "timestamp")
        .group_by(["uid", "item_id"])
        .agg(pl.col("timestamp").min().alias("t_any"),
             pl.col("timestamp").filter(pl.col("timestamp") < TEST0).min().alias("t_pre"))
        .collect(engine="streaming"))

j = lk.join(ls, on=["uid", "item_id"], how="left")
after  = j.filter(pl.col("t_pre") <= pl.col("t_like"))                      # слушал, потом лайк
before = j.filter((pl.col("t_pre") > pl.col("t_like")) | (pl.col("t_pre").is_null() & pl.col("t_any").is_not_null()))
never  = j.filter(pl.col("t_any").is_null())
tot = j.height
print("\n=== из чего состоят лайки (пары, лайк до начала теста) ===")
for name, d in [("лайк ПОСЛЕ первого прослушивания (реакция на звук)", after),
                ("лайк РАНЬШЕ прослушивания, но прослушивание есть", before),
                ("лайк БЕЗ прослушивания вообще", never)]:
    print("  %-52s %7d (%5.1f%%)  органических %4.1f%%"
          % (name, d.height, 100*d.height/tot, 100*d["org_like"].mean()))

print("\n=== у 'лайк раньше прослушивания': когда доходит до прослушивания ===")
b = before.with_columns(((pl.col("t_any") - pl.col("t_like"))/86400).alias("delay"))
for q in [.25, .5, .75, .9]:
    print("   q%02d = %.1f суток" % (int(q*100), b["delay"].quantile(q)))
print("   доля дошедших в первые сутки: %.1f%%" % (100*b.filter(pl.col("delay") <= 1).height/b.height))
print("   из них прослушивание попадает уже в тест: %d пар (%.1f%%)"
      % (b.filter(pl.col("t_any") >= TEST0).height, 100*b.filter(pl.col("t_any") >= TEST0).height/b.height))

print("\n=== то же внутри каталога (10 000 треков) ===")
jc = j.filter(pl.col("item_id").is_in(list(cat)))
ac = jc.filter(pl.col("t_pre") <= pl.col("t_like"))
print("  пар с лайком в каталоге: %d, из них лайк после прослушивания %.1f%%"
      % (jc.height, 100*ac.height/jc.height))
