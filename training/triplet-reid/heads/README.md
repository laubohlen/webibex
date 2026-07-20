Per-file porting provenance for the `tf.contrib.slim` -> `tf_slim` port
(TF1 -> TF2 migration, see `docs/tf1-to-tf2-migration-plan.md` at the
webibex repo root for the full diagnostic trail).

`fc1024.py` and `__init__.py` copied verbatim from
`triplet-reid_v2adapted/heads/` — confirmed complete port
(`tf.contrib.slim` -> `tf_slim`, `tf.GraphKeys` -> `tf.compat.v1.GraphKeys`,
`tf.orthogonal_initializer` -> `tf.keras.initializers.Orthogonal()`).

Necessary, minimal dependency of `export_saved_model.py`: the embedding
graph can't be built without a head module. Training config confirms
`head_name="fc1024"`, `embedding_dim=128`.

`direct.py`, `direct_normalize.py`, `fc1024_normalize.py` were dropped —
unused by the production single-horn-crop chip path (see `HEAD_CHOICES` in
`__init__.py`, which still lists them for the unported training scripts;
only `fc1024` is wired into the export pipeline).
