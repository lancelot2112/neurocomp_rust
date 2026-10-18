# Rung log recovery, runs 29-33

Through run 33 the rung log wrote an island's elite one generation after its pass, ranked on the
next rung. Each dump was rescored on the rung it passed (24 fresh worlds, ISL_SIGNAL=24
ISL_SIGNAL_EACH=1, the run's own settings, the current binary). Kept: the dumped genomes scoring at
least 0.8 of the goal, best first; failing that, the best genome of the run's final population
on that rung (a relative, not the passer); otherwise none. The generalist's own-ladder dumps
were written in the right generation and are kept as they are. `rungs-fixed.txt` in each run.

| run | island | rung passed | generation | held out | goal | dump scores | kept |
|---|---|---|---|---|---|---|---|
| 29 | reading | babble | 6 | 814 | 768 | 917 | 1 of the dump (best 917) |
| 29 | shapes | clean | 17 | 159 | 150 | 85 | a relative from the final population (397) |
| 29 | shapes | blur1 | 17 | 159 | 150 | 219 | 1 of the dump (best 219) |
| 29 | reading | echo2q | 18 | 174 | 150 | 466 | 1 of the dump (best 466) |
| 29 | shapes | blur2 | 19 | 156 | 150 | 167 | 1 of the dump (best 167) |
| 29 | shapes | shift8 | 23 | 102 | 120 | 87 | a relative from the final population (295) |
| 29 | shapes | shift64 | 23 | 102 | 100 | 129 | 1 of the dump (best 129) |
| 29 | reading | echo2m8 | 46 | 155 | 150 | 291 | 1 of the dump (best 291) |
| 29 | reading | echo2m4 | 48 | 195 | 150 | 63 | none (final population best 38) |
| 29 | switching | respond | 51 | 118 | 100 | 322 | 1 of the dump (best 322) |
| 29 | screen | glow-150 | 52 | 164 | 150 | 178 | 1 of the dump (best 178) |
| 29 | switching | tell | 53 | 126 | 100 | 87 | 1 of the dump (best 87) |
| 29 | recall | respond | 58 | 154 | 100 | 156 | 1 of the dump (best 156) |
| 29 | recall | lag1 | 73 | 154 | 150 | 184 | 1 of the dump (best 184) |
| 29 | recall | lag2 | 73 | 154 | 150 | 172 | 1 of the dump (best 172) |
| 30 | reading | babble | 11 | 1024 | 768 | 1024 | 1 of the dump (best 1024) |
| 30 | shapes | clean | 12 | 240 | 150 | 240 | 1 of the dump (best 240) |
| 30 | shapes | blur1 | 13 | 259 | 150 | 343 | 1 of the dump (best 343) |
| 30 | shapes | blur2 | 14 | 150 | 150 | 363 | 1 of the dump (best 363) |
| 30 | shapes | shift8 | 14 | 150 | 120 | 188 | 1 of the dump (best 188) |
| 30 | shapes | shift64 | 17 | 101 | 100 | 114 | 1 of the dump (best 114) |
| 30 | screen | glow-150 | 19 | 259 | 150 | 295 | 1 of the dump (best 295) |
| 30 | switching | respond | 28 | 209 | 100 | 42 | a relative from the final population (842) |
| 30 | switching | tell | 28 | 209 | 100 | 213 | 1 of the dump (best 213) |
| 30 | recall | respond | 33 | 610 | 100 | 380 | 1 of the dump (best 380) |
| 30 | recall | lag1 | 34 | 921 | 150 | 945 | 1 of the dump (best 945) |
| 30 | recall | lag2 | 35 | 731 | 150 | 975 | 1 of the dump (best 975) |
| 30 | recall | lag3 | 35 | 731 | 150 | 574 | 1 of the dump (best 574) |
| 30 | screen | glow | 41 | 307 | 300 | 313 | 1 of the dump (best 313) |
| 30 | maze | feel-100 | 49 | 178 | 100 | 210 | 1 of the dump (best 210) |
| 30 | maze | feel | 49 | 178 | 160 | 209 | 1 of the dump (best 209) |
| 30 | recall | lag4 | 62 | 152 | 150 | 20 | a relative from the final population (185) |
| 30 | recall | lag5 | 154 | 276 | 150 | 100 | a relative from the final population (169) |
| 30 | recall | lag8 | 208 | 162 | 150 | 0 | none (final population best 14) |
| 30 | prediction | cycle1-100 | 214 | 115 | 100 | 48 | a relative from the final population (203) |
| 30 | shapes | feel-100 | 229 | 139 | 100 | 238 | 1 of the dump (best 238) |
| 30 | shapes | feel | 233 | 339 | 160 | 238 | 1 of the dump (best 238) |
| 30 | shapes | repeat | 233 | 339 | 300 | 282 | 1 of the dump (best 282) |
| 30 | prediction | cycle1 | 254 | 207 | 200 | 246 | 1 of the dump (best 246) |
| 31 | reading | babble | 10 | 1013 | 768 | 1024 | 1 of the dump (best 1024) |
| 31 | switching | respond | 18 | 121 | 100 | 234 | 1 of the dump (best 234) |
| 31 | screen | glow-150 | 19 | 195 | 150 | 239 | 1 of the dump (best 239) |
| 31 | reading | echo2q | 23 | 182 | 150 | 351 | 1 of the dump (best 351) |
| 31 | screen | glow | 46 | 338 | 300 | 365 | 1 of the dump (best 365) |
| 31 | recall | respond | 49 | 109 | 100 | 124 | 1 of the dump (best 124) |
| 31 | recall | lag1 | 60 | 215 | 150 | 274 | 1 of the dump (best 274) |
| 31 | recall | lag2 | 60 | 215 | 150 | 257 | 1 of the dump (best 257) |
| 31 | shapes | clean | 66 | 151 | 150 | 153 | 1 of the dump (best 153) |
| 31 | switching | tell | 78 | 107 | 100 | 427 | 1 of the dump (best 427) |
| 31 | switching | feel-100 | 87 | 219 | 100 | 0 | none (final population best 4) |
| 31 | switching | feel | 97 | 190 | 160 | 0 | none (final population best 4) |
| 31 | recall | lag3 | 104 | 202 | 150 | 96 | a relative from the final population (285) |
| 31 | shapes | blur1 | 148 | 157 | 150 | 294 | 1 of the dump (best 294) |
| 31 | recall | lag4 | 188 | 154 | 150 | 0 | a relative from the final population (204) |
| 31 | maze | feel-100 | 214 | 115 | 100 | 151 | 1 of the dump (best 151) |
| 31 | recall | lag5 | 278 | 257 | 150 | 0 | a relative from the final population (142) |
| 31 | recall | lag8 | 280 | 156 | 150 | 0 | a relative from the final population (117, weak) |
| 31 | prediction | cycle1-100 | 282 | 102 | 100 | 129 | 1 of the dump (best 129) |
| 31 | maze | feel | 282 | 180 | 160 | 182 | 1 of the dump (best 182) |
| 31 | prediction | cycle1 | 294 | 213 | 200 | 224 | 1 of the dump (best 224) |
| 31 | prediction | cycle2 | 301 | 246 | 200 | 275 | 1 of the dump (best 275) |
| 31 | recall | lag16 | 304 | 158 | 150 | -8 | a relative from the final population (120) |
| 31 | prediction | cycle3 | 313 | 250 | 200 | 465 | 1 of the dump (best 465) |
| 31 | prediction | cycle4 | 313 | 250 | 200 | 331 | 1 of the dump (best 331) |
| 32 | reading | babble | 9 | 953 | 768 | 1003 1003 981 1003 1003 1003 1024 1024 | 8 of the dump (best 1024) |
| 32 | switching | respond | 11 | 288 | 100 | 407 407 426 386 276 276 185 225 | 8 of the dump (best 426) |
| 32 | screen | glow-150 | 15 | 158 | 150 | 375 242 242 242 319 226 226 226 | 8 of the dump (best 375) |
| 32 | screen | glow | 20 | 303 | 300 | 405 405 451 358 372 357 310 170 | 7 of the dump (best 451) |
| 32 | recall | respond | 44 | 114 | 100 | 130 126 104 92 92 118 83 55 | 7 of the dump (best 130) |
| 32 | reading | echo2q | 64 | 205 | 150 | 402 400 402 402 402 402 412 10 | 7 of the dump (best 412) |
| 32 | recall | lag1 | 71 | 187 | 150 | 254 193 219 191 232 250 83 115 | 6 of the dump (best 254) |
| 32 | recall | lag2 | 71 | 187 | 150 | 84 72 95 267 178 225 281 272 | 5 of the dump (best 281) |
| 32 | shapes | clean | 90 | 166 | 150 | 87 87 87 5 37 37 148 50 | 1 of the dump (best 148) |
| 32 | recall | lag3 | 103 | 167 | 150 | 120 134 128 81 165 84 84 205 | 5 of the dump (best 205) |
| 32 | shapes | blur1 | 130 | 219 | 150 | 0 0 9 212 1 332 332 332 | 4 of the dump (best 332) |
| 32 | recall | lag4 | 217 | 238 | 150 | 423 518 506 480 -227 446 -263 -261 | 5 of the dump (best 518) |
| 32 | recall | lag5 | 224 | 150 | 150 | -211 -211 -217 -156 -249 -235 -229 397 | 1 of the dump (best 397) |
| 32 | shapes | blur2 | 321 | 218 | 150 | 148 148 148 148 148 226 265 265 | 8 of the dump (best 265) |
| 32 | shapes | shift8 | 322 | 108 | 120 | 137 137 137 137 239 0 17 246 | 6 of the dump (best 246) |
| 32 | shapes | shift64 | 322 | 108 | 100 | 130 130 130 94 0 16 4 122 | 5 of the dump (best 130) |
| 32 | recall | lag8 | 328 | 164 | 150 | -7 -130 -115 245 255 87 496 523 | 4 of the dump (best 523) |
| 32 | recall | lag16 | 408 | 243 | 150 | 0 0 0 0 0 0 0 0 | none (final population best 8) |
| 32 | maze | feel-300 | 473 | 335 | 300 | 323 319 308 296 294 292 284 278 | 8 of the dump (best 323) |
| 33 | reading | babble | 5 | 836 | 768 | 1024 1024 1024 1024 1024 917 917 960 | 8 of the dump (best 1024) |
| 33 | switching | respond | 10 | 230 | 100 | 407 468 468 521 387 256 256 256 | 8 of the dump (best 521) |
| 33 | recall | respond | 20 | 131 | 100 | 194 214 214 214 214 186 198 121 | 8 of the dump (best 214) |
| 33 | shapes | clean | 21 | 383 | 150 | 469 469 469 469 469 469 469 427 | 8 of the dump (best 469) |
| 33 | recall | lag1 | 24 | 271 | 150 | 308 338 308 342 355 99 266 158 | 7 of the dump (best 355) |
| 33 | recall | lag2 | 24 | 271 | 150 | 387 400 324 293 282 234 300 377 | 8 of the dump (best 400) |
| 33 | reading | echo2q | 32 | 167 | 150 | 412 377 377 2 2 2 -19 -18 | 3 of the dump (best 412) |
| 33 | screen | glow-150 | 34 | 150 | 150 | 79 101 101 189 189 176 176 42 | 4 of the dump (best 189) |
| 33 | recall | lag3 | 44 | 150 | 150 | 0 0 0 53 128 128 2 55 | 2 of the dump (best 128) |
| 33 | screen | glow | 93 | 328 | 300 | 317 301 296 173 269 235 217 240 | 5 of the dump (best 317) |
| 33 | recall | lag4 | 233 | 199 | 150 | 0 0 0 0 0 0 -1 -1 | none (final population best 21) |
| 33 | recall | lag5 | 264 | 174 | 150 | 2 2 2 2 0 0 0 2 | none (final population best 26) |
| 33 | maze | feel-300 | 324 | 331 | 300 | 253 196 264 237 296 296 255 255 | 6 of the dump (best 296) |
| 33 | prediction | cycle1-100 | 415 | 147 | 100 | 192 192 192 192 1 2 16 22 | 4 of the dump (best 192) |
| 33 | recall | lag8 | 444 | 174 | 150 | 0 0 0 0 0 0 0 0 | none (final population best 18) |
| 33 | recall | lag16 | 468 | 246 | 150 | 0 0 0 0 0 0 0 0 | none (final population best 18) |
| 33 | prediction | cycle1 | 497 | 286 | 200 | 226 123 160 163 163 86 356 282 | 6 of the dump (best 356) |
| 33 | prediction | cycle2 | 518 | 287 | 200 | 5 31 31 -28 -28 -6 -46 7 | none (final population best 18) |
| 33 | selection | cued | 565 | 641 | 600 | 622 622 592 592 592 592 534 527 | 8 of the dump (best 622) |
| 33 | selection | uncued | 567 | 665 | 600 | 581 597 584 547 371 446 408 291 | 4 of the dump (best 597) |
| 33 | screen | torch | 933 | 338 | 300 | 188 206 269 269 224 124 168 168 | 2 of the dump (best 269) |
| 33 | screen | two | 1030 | 310 | 300 | -21 -21 -21 -21 -21 -21 -21 -21 | a relative from the final population (510) |
