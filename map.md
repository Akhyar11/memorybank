Ya. Kalau arahnya **bukan SNN**, saya justru melihat ide ini lebih menarik kalau kita jadikan sebuah **Tiny Memory-Augmented MoE Language Model** sekitar **50M parameter**.

Dan menurut saya, **jangan mulai dari general-purpose language model**. Untuk 50M parameter, tugas yang paling tepat adalah **conversational memory / personalized dialogue**, karena di situlah keunggulan Memory Layer bisa diukur secara jelas.

Ada literatur yang sangat dekat dengan idemu: Key-Value Memory Networks memang melakukan query terhadap pasangan key-value dan membaca memory yang relevan; bahkan penelitian lanjutannya menggunakan *query updating* untuk multi-hop retrieval. ([ACL Anthology][1]) Yang menarik, riset 2026 juga mulai menggabungkan **conditional memory + MoE**, walaupun pendekatannya berbeda dari memory percakapan mutable yang kamu bayangkan. ([ACL Anthology][2])

---

# Rancangan Penelitian

## Judul sementara

Saya menyarankan judul:

> **Memory-Augmented Mixture-of-Experts Language Model for Persistent Conversational Context**

Atau kalau ingin lebih kuat secara akademik:

> **A Tiny Memory-Augmented Mixture-of-Experts Language Model with Dynamic Top-K Memory Retrieval for Persistent Conversational Context**

Saya lebih suka judul kedua untuk paper.

---

# 1. Masalah yang ingin kita selesaikan

LLM konvensional memiliki masalah fundamental ketika menangani informasi percakapan yang harus dipertahankan.

Contohnya:

```text
A: Nama saya Akhyar.
B: Halo Akhyar.

... 1000 turn kemudian ...

A: Siapa nama saya?
B: Nama kamu Akhyar.
```

Model Transformer konvensional pada dasarnya memperoleh informasi tersebut dengan memasukkan kembali conversation history ke dalam context.

Artinya:

$$
x_1,x_2,...,x_t
$$

harus tetap tersedia di context.

Akibatnya muncul tiga masalah:

### Masalah 1 — Context window

Semakin panjang percakapan:

$$
O(T^2)
$$

pada self-attention menjadi mahal.

---

### Masalah 2 — Informasi lama harus terus dibawa

Informasi:

```text
nama = Akhyar
umur = 22
tinggal = Solo
suka = apel
```

sebenarnya hanya membutuhkan beberapa memory slot.

Tetapi model harus membawa seluruh:

```text
conversation history
```

untuk mengaksesnya.

---

### Masalah 3 — Model tidak mempunyai memory yang benar-benar mutable

Kita ingin:

```text
nama saya A
```

kemudian:

```text
nama saya B
```

maka memory dapat berubah:

```text
name → A
```

menjadi:

```text
name → B
```

tanpa retraining parameter model.

Ini yang menurut saya menjadi **core research problem**.

---

# 2. Solusi yang kita tawarkan

Kita buat:

# **Memory-Augmented MoE (MA-MoE)**

dengan tiga komponen utama:

```text
                 INPUT
                   │
                   ▼
            ┌─────────────┐
            │ Tiny LLM    │
            │ Transformer │
            └──────┬──────┘
                   │
             hidden state
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
    ┌───────────┐      ┌───────────┐
    │ MoE       │      │ Memory    │
    │ Router    │      │ Router    │
    └─────┬─────┘      └─────┬─────┘
          │                  │
       Top-2 Experts       Top-K
          │                  │
          ▼                  ▼
    ┌─────────────┐    ┌─────────────┐
    │ Experts     │    │ Memory Bank │
    │ E1 E2 E3 E4 │    │ K → V       │
    └──────┬──────┘    └──────┬──────┘
           │                  │
           └────────┬─────────┘
                    ▼
             Memory Fusion
                    │
                    ▼
               Decoder
                    │
                    ▼
                  Output
```

---

# 3. Perbedaan MoE dan Memory

Ini harus dibuat sangat jelas dalam paper.

### MoE

MoE menjawab:

> **"Bagaimana saya memproses informasi ini?"**

Router:

$$
r(x) \rightarrow E_i
$$

Contohnya:

```text
math → Expert 2
language → Expert 1
reasoning → Expert 4
code → Expert 3
```

---

### Memory

Memory menjawab:

> **"Apa yang pernah saya ketahui dari interaksi sebelumnya?"**

Query:

$$
q=f_\theta(h_t)
$$

kemudian:

$$
M(q)=TopK(q,K)
$$

Contoh:

```text
query:
"siapa nama saya?"

      ↓

Memory

name → Akhyar
age → 22
city → Solo
favorite_food → apple

      ↓

Top-K

name → Akhyar
```

---

# 4. Memory Bank

Saya menyarankan struktur:

$$
M=\{(k_i,v_i,s_i,t_i,c_i)\}_{i=1}^{N}
$$

dengan:

| Komponen | Fungsi       |
| -------- | ------------ |
| \(k_i\)  | memory key   |
| \(v_i\)  | memory value |
| \(s_i\)  | strength     |
| \(t_i\)  | timestamp    |
| \(c_i\)  | confidence   |

Contoh:

```text
Memory #182

KEY:
"user name"

VALUE:
"Akhyar"

STRENGTH:
0.93

CONFIDENCE:
0.98

TIMESTAMP:
172
```

---

# 5. Query mechanism

Hidden state:

$$
h_t\in R^d
$$

diubah menjadi query:

$$
q=W_qh_t
$$

Memory key:

$$
k_i=W_kk_i
$$

similarity:

$$
s_i=\frac{q^Tk_i}{\sqrt d}
$$

Kemudian:

$$
I=TopK(s_1,...,s_N)
$$

Misalnya:

```text
Memory size = 100,000
K = 4

Query
  ↓
100,000 memories
  ↓
Top 4
```

Yang masuk ke model hanya empat memory.

---

# 6. Memory read

Setelah Top-K:

$$
\alpha_i=
\frac{e^{s_i}}
{\sum_{j\in TopK}e^{s_j}}
$$

kemudian:

$$
r=\sum_{i\in TopK}\alpha_i v_i
$$

Jadi:

```text
Query
  ↓
Top-K
  ↓
Weighted memory
  ↓
r
```

Kemudian fusion:

$$
h'_t=h_t+g_t\odot W_rr
$$

dengan:

$$
g_t=\sigma(W_gh_t)
$$

Gate ini penting.

Karena model harus bisa mengatakan:

> "Memory ini tidak relevan."

---

# 7. Memory WRITE

Ini bagian yang menurut saya menjadi kontribusi utama.

Tidak setiap token boleh ditulis.

Kita buat:

$$
p_{write}=\sigma(W_wh_t)
$$

Jika:

$$
p_{write}>\tau
$$

maka:

```text
WRITE
```

Misalnya:

```text
"Nama saya Akhyar."

        ↓

write probability = 0.97

        ↓

Memory

key = "user name"
value = "Akhyar"
```

Sedangkan:

```text
"Hari ini cuacanya bagus."

write probability = 0.03
```

tidak ditulis.

---

# 8. Memory UPDATE

Ini bagian yang sangat penting.

Misalnya:

```text
Memory:

user_name → Akhyar
```

Kemudian:

```text
Nama saya Budi.
```

Model melakukan query:

```text
"user name"
```

dan menemukan:

```text
user_name → Akhyar
```

kemudian update:

$$
v_{new}=
(1-\lambda)v_{old}
+\lambda v_{candidate}
$$

atau bahkan lebih sederhana:

```text
old:
Akhyar

new:
Budi
```

Untuk fakta tertentu kita dapat menggunakan **replacement policy**.

Untuk fakta yang sifatnya kumpulan:

```text
user likes:

apple
banana
coffee
```

kita menggunakan append/update.

Jadi memory memiliki **memory semantics**, bukan sekadar vector database.

---

# 9. MoE + Memory

Sekarang bagian yang menurut saya paling menarik.

Kita tidak hanya memiliki:

```text
MoE Router
```

tetapi:

```text
                  Hidden State
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        MoE Router          Memory Router
             │                   │
          Top-2               Top-K
             │                   │
             ▼                   ▼
         Experts             Memories
             │                   │
             └─────────┬─────────┘
                       ▼
                     Fusion
```

Sehingga:

### MoE

menyediakan **computational specialization**.

### Memory

menyediakan **information persistence**.

Ini memberikan hipotesis yang bagus:

> **MoE meningkatkan kapasitas komputasi secara conditional, sedangkan Memory Layer meningkatkan kapasitas konteks secara persistent tanpa meningkatkan parameter trainable model secara proporsional.**

---

# 10. Model 50M

Untuk Kaggle T4, saya justru tidak akan membuat model tepat 50M dari awal.

Saya akan target:

> **40–55M trainable parameters**

Misalnya:

```text
Vocabulary       32K
d_model          384
Layers           8
Attention heads  6
FFN              1024
Experts          4
Top-K experts    2
Memory dim       384
Memory slots     100K
Memory Top-K     4
Context          512 / 1024
```

Namun ada satu trik penting:

## Memory tidak dihitung sebagai parameter model

Memory:

```text
100,000 × 384
```

memang secara fisik besar:

$$
38.4M
$$

vectors.

Tetapi ia bukan **trainable parameters**.

Jadi paper harus membedakan:

```text
Trainable Parameters
+
External Mutable Memory Capacity
```

Ini justru bisa menjadi salah satu kontribusi eksperimen.

---

# 11. Tugas yang saya rekomendasikan

Saya **tidak menyarankan translation sebagai tugas utama**.

Untuk ide ini, tugas terbaik adalah:

# **Conversational Memory / Personalized Dialogue**

Kenapa?

Karena kita bisa membuat eksperimen yang sangat jelas.

### Task 1 — Fact Recall

```text
User:
Nama saya Akhyar.

Model:
Halo Akhyar.

...

User:
Siapa nama saya?

Model:
Nama kamu Akhyar.
```

---

### Task 2 — Multi-fact Recall

```text
User:
Nama saya Akhyar.
Saya berumur 22 tahun.
Saya tinggal di Solo.
Saya suka apel.

...

User:
Berapa umur saya?

→ 22 tahun
```

---

### Task 3 — Memory Update

```text
User:
Saya tinggal di Solo.

...

User:
Sekarang saya tinggal di Jakarta.

...

User:
Saya tinggal di mana?

→ Jakarta
```

---

### Task 4 — Distractor Resistance

Ini penting.

```text
User:
Nama saya Akhyar.

[random 100 turns]

User:
Saya suka apel.

[random 100 turns]

User:
Apa nama saya?
```

Model harus mengambil:

```text
name → Akhyar
```

tanpa membaca 200 turn sebelumnya.

---

# 12. Dataset

Saya bahkan menyarankan **dataset synthetic + real NLP benchmark**.

### Synthetic Conversational Memory Dataset

Kita generate:

```text
PERSON
NAME
AGE
CITY
JOB
HOBBY
FOOD
PET
FAVORITE_COLOR
```

Contoh:

```text
User:
Nama saya Rina.

Assistant:
Halo Rina.

User:
Saya tinggal di Bandung.

Assistant:
Baik.

User:
Saya suka pizza.

Assistant:
Menarik.

...

Question:
Siapa nama saya?
```

Ini memungkinkan kita mengontrol:

* memory distance
* number of distractors
* number of facts
* update frequency
* conflicting facts
* memory size

---

# 13. Eksperimen utama

Kita harus memiliki **ablation study**.

### Model A

```text
Tiny Transformer
```

### Model B

```text
Tiny Transformer + MoE
```

### Model C

```text
Tiny Transformer + Memory
```

### Model D

```text
Tiny Transformer + MoE + Memory
```

Ini sangat penting.

Karena kita bisa menjawab:

> Apakah improvement berasal dari MoE?

dan:

> Apakah improvement berasal dari Memory?

dan yang paling penting:

> Apakah kombinasi MoE + Memory memberikan keuntungan tambahan?

---

# 14. Rumusan masalah

Saya menyarankan **3 rumusan masalah**.

### RQ1

> **Bagaimana merancang Memory Layer yang mampu melakukan retrieval Top-K terhadap representasi percakapan serta melakukan penambahan dan pembaruan memory secara dinamis tanpa memerlukan penyimpanan ulang seluruh conversation history pada context window?**

---

### RQ2

> **Sejauh mana integrasi Dynamic Memory Layer dapat meningkatkan kemampuan model bahasa berukuran kecil dalam mempertahankan dan mengambil kembali informasi percakapan jangka panjang dibandingkan model bahasa tanpa memory?**

---

### RQ3

> **Apakah kombinasi Dynamic Memory Layer dengan Mixture-of-Experts dapat meningkatkan kemampuan persistent conversational context secara lebih efektif dibandingkan penggunaan MoE atau Memory Layer secara terpisah pada jumlah parameter trainable yang relatif sama?**

Menurut saya **RQ3 adalah yang paling kuat** untuk paper.

---

# 15. Hipotesis

### H1

$$
Memory > NoMemory
$$

pada long-range conversational recall.

### H2

$$
MoE+Memory > MoE
$$

pada persistent context.

### H3

Memory mampu mempertahankan informasi dengan jumlah context yang lebih kecil.

### H4

Dynamic update memungkinkan model menangani perubahan fakta tanpa retraining.

---

# 16. Metric

Jangan hanya pakai perplexity.

Kita butuh:

### Memory Recall Accuracy

$$
MRA=
\frac{correct\ retrieved\ memories}
{total\ queries}
$$

### Answer Accuracy

Apakah jawaban benar.

### Memory Update Accuracy

```text
old fact → new fact
```

berapa persen berhasil.

### Context Compression Ratio

Misalnya:

```text
Traditional:
10,000 tokens conversation

Memory:
200 memory tokens
```

$$
CCR=\frac{10000}{200}=50\times
$$

Ini metrik yang **sangat bagus untuk paper**.

### Long-range Recall

Ukur berdasarkan:

```text
10 turns
50 turns
100 turns
500 turns
1000 turns
```

Kita bisa membuat grafik:

```text
Accuracy
│
│\
│ \
│  \       Transformer
│   \________
│
│────────────── Memory Model
│
└──────────────────
     Distance
```

---

# 17. Yang paling menarik: Memory tidak perlu masuk context

Ini harus menjadi prinsip desain.

Model konvensional:

```text
Conversation
   ↓
Context
   ↓
Transformer
```

Sedangkan model kita:

```text
Conversation
      ↓
 Memory Writer
      ↓
 ┌─────────────┐
 │ Memory Bank │
 └──────┬──────┘
        │
       query
        │
        ▼
      Top-K
        │
        ▼
   Tiny MoE LLM
```

Jadi **conversation history bukan lagi storage utama**.

Memory Bank menjadi storage utama untuk informasi yang dianggap penting.

---

# 18. Kenapa MoE diperlukan?

Ini pertanyaan yang mungkin akan muncul dari reviewer:

> "Kalau tujuan utamanya memory, kenapa pakai MoE?"

Jawabannya:

Karena memory dan MoE memiliki fungsi yang berbeda.

```text
                 MODEL
                   │
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
   MEMORY                    MoE
       │                       │
  "What do I know?"      "How do I process?"
       │                       │
 Persistent information    Conditional compute
```

Dengan kata lain:

> **Memory menyediakan persistent information capacity, sedangkan MoE menyediakan conditional computation capacity.**

Keduanya orthogonal.

Ini juga relevan dengan penelitian terbaru yang mulai mengeksplorasi memory sebagai dimensi sparsity tambahan di samping MoE. ([ACL Anthology][2])

---

# 19. Posisi penelitianmu

Saya akan memosisikannya bukan sebagai:

> "Kami membuat LLM kecil."

Tetapi:

> **"Kami mengeksplorasi apakah persistent mutable memory dapat menggantikan sebagian kebutuhan context history pada tiny language models, dan apakah conditional computation melalui MoE dapat melengkapi memory tersebut."**

Itu jauh lebih menarik.

Dan ada perkembangan riset terbaru yang membuat positioning ini semakin relevan: paper ACL 2026 *Conditional Memory via Scalable Lookup* secara eksplisit memandang memory sebagai dimensi sparsity lain selain MoE. Namun memory mereka berbasis indexed embedding dengan local n-gram lookup, sehingga **berbeda dengan mutable conversational memory yang kamu usulkan**. ([ACL Anthology][2])

---

## Rekomendasi final saya

Kalau targetmu adalah **train full di Kaggle T4**, saya akan membuat versi pertama seperti ini:

```text
                MA-MoE 50M
                   │
       ┌───────────┴───────────┐
       │                       │
   Tiny Transformer          Memory
       │                       │
    8 layers              100K slots
       │                       │
   4 Experts                  Top-4
       │                       │
       └───────────┬───────────┘
                   │
                 Fusion
                   │
                Decoder
```

**Tugas utama:** conversational persistent memory.

**Dataset awal:** synthetic long-context dialogue → kemudian benchmark NLP conversational memory.

**Kontribusi utama:**

1. **Dynamic Memory Layer**
2. **Top-K semantic memory retrieval**
3. **Online memory write**
4. **Online memory update**
5. **Memory-aware MoE routing**
6. **Persistent context tanpa memasukkan conversation history penuh**
7. **Evaluasi memory distance + context compression**

Dan saya akan **menghindari menyebutnya RAG** dalam paper. Lebih tepat menyebutnya **Dynamic Neural Memory / Persistent Conversational Memory**, karena memory tersebut bukan corpus eksternal statis yang di-retrieve, tetapi **state yang dibentuk dan dimutasi oleh model selama interaksi**.

Kalau kita lanjut ke tahap implementasi, saya menyarankan kita **jangan langsung coding model 50M**. Kita desain dulu spesifikasi matematis lengkap untuk **Memory Layer + MoE Router + Memory Write/Update policy**, lalu hitung parameter satu per satu supaya model benar-benar bisa dilatih full di Kaggle T4.

[1]: https://aclanthology.org/D16-1147/?utm_source=chatgpt.com "Key-Value Memory Networks for Directly Reading Documents - ACL Anthology"
[2]: https://aclanthology.org/2026.acl-long.226/?utm_source=chatgpt.com "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models - ACL Anthology"

---

# 20. Final Architecture Lock: UGMB-MoE

*(Kesimpulan dari diskusi desain akhir)*

Arsitektur final telah dikunci dengan konsep **Utterance-Level Gated Memory Bank (UGMB)** yang digabungkan dengan **MoE**, menjadi **UGMB-MoE**.

## Prinsip Utama
1. **Evaluasi Utterance-Level**: Tidak ada operasi *read/write* di setiap token. *Memory Gate* hanya dievaluasi satu kali pada akhir sebuah giliran bicara (*utterance*) menggunakan representasi *hidden state* dari token `<EOS>` ($h_{EOS}$).
2. **3-Way Memory Gate**: $g = Softmax(W_g h_{EOS})$, memproduksi probabilitas `[NONE, READ, WRITE]`.
3. **Mekanisme READ**: Jika $P_{read} > \tau$, model memproduksi *query* $q = W_q h_{EOS}$, menghitung *similarity* $s_i = q^T k_i$, dan mengambil $TopK(k_i)$ (direkomendasikan $K=4$).
4. **Mekanisme WRITE & UPDATE**: Jika $P_{write} > \tau$, $h_{EOS}$ di-*encode* menjadi *key-value*. Sebelum disimpan, sistem melakukan *similarity search* ke *Memory Bank*. Jika ada kecocokan (*match*), maka nilai yang lama akan di-**UPDATE** menjadi state terbaru, bukan sekadar di-*insert* sebagai entri ganda.
5. **Differentiable Memory Bank**: Memory direpresentasikan sebagai entitas $M = \{(k_i,v_i,t_i,m_i)\}_{i=1}^{N}$ dengan *learned keys/values*, membuatnya berbeda secara fundamental dengan basis data statis seperti FAISS.
6. **Separation of Concerns (Memory vs Compute)**: *Memory Router* (*Read/Write/None*) beroperasi sepenuhnya terpisah dari *MoE Router* (*Expert 1/2/3/4*). Keduanya menyelesaikan dua limitasi independen (*persistence* vs *conditional compute*).

## Diagram Alur Arsitektur

```text
Input Tokens
    │
    ▼
Embedding
32K × 384
    │
    ▼
┌─────────────────────────────┐
│ Transformer Block × 8       │
│                             │
│ RMSNorm                     │
│   ↓                         │
│ Causal Self-Attention       │
│ 6 heads × 64                │
│   ↓                         │
│ Residual                    │
│   ↓                         │
│ RMSNorm                     │
│   ↓                         │
│ MoE FFN                     │
│ 4 Experts → Top-2 Routing   │
│ Expert FFN: 384 → 1536 →384 │
│   ↓                         │
│ Residual                    │
└──────────────┬──────────────┘
               │
               ▼
             h_EOS
               │
               ▼
      ┌──────────────────┐
      │ Memory Controller│
      │ Softmax: 3-way   │
      └────────┬─────────┘
               │
        ┌──────┼──────┐
        ▼      ▼      ▼
      NONE    READ   WRITE
                │      │
                ▼      ▼
             Query   Key/Value
                │      │
                ▼      ▼
             Top-K   Existing
             K=4    Memory Search
                       │
                  ┌────┴────┐
                  ▼         ▼
                INSERT    UPDATE
                       │
                       ▼
                 Memory Bank
                 10K slots
                       │
                       ▼
                Memory Fusion
                       │
                       ▼
                 LM Head
                 (tied embedding)
                       │
                       ▼
                    Output
```

## Konfigurasi Final MAMoE-50

| Komponen                |                   Nilai |
| ----------------------- | ----------------------: |
| Embedding               |        **di luar 50M**  |
| Parameters (Total)      |                **~52M** |
| Active Params per token |                 **~5M** |
| Vocabulary              |                     32K |
| Hidden size             |                     256 |
| Layers                  |                       8 |
| Attention heads         |                       4 |
| Head dimension          |                      64 |
| FFN dimension           |                     512 |
| MoE experts             |                      16 |
| Expert routing          |                   Top-1 |
| Position                |                    RoPE |
| Normalization           |                 RMSNorm |
| Activation              |                  SwiGLU |
| Memory capacity         | 10K (di luar parameter) |
| Memory Top-K            |                       4 |
| Memory operation        |   READ / WRITE / UPDATE |
| Memory gating           | **Utterance/EOS-level** |
| Memory query            |    **1× per utterance** |
| RAG                     |               **Tidak** |
| Per-token memory access |               **Tidak** |

**Catatan Khusus Dataset**: Dataset yang dihasilkan oleh script `memorybench_groq_generator.py` (5.000 JSON episodes) kompatibel dan *perfectly aligned* dengan desain ini, karena secara *native* mensimulasikan aliran fakta tunggal, pengujian ingatan, transisi mulus, distraktor, dan pembaruan (*updates*) memori secara natural.
