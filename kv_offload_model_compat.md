# KV Offload (external KV connector) 모델 아키텍처 호환성 정리

기준: 벤더링된 vLLM 체크아웃 (upstream v0.20.1rc1.dev57, 2026-04-29) + `SimpleCPUOffloadConnector` 기반 SSLO offload tier. 코드 조사만으로 작성 (2026-08-04); 실증 근거는 WORKLOG 2026-08-04 항목 참조.

## 판정 기준 (코드 근거)

- **G1 — mamba/linear-attn 그룹 + `mamba_cache_mode="align"` → 크래시.**
  `scheduler.py:397-401`에서 `need_mamba_block_aligned_split = has_mamba_layers and mamba_cache_mode=="align"`. `_mamba_block_aligned_split()` 첫 줄이 `assert num_external_computed_tokens == 0` (`scheduler.py:452`). external tokens를 실제로 넘기는 waiting 스케줄 경로(`scheduler.py:856-864`, SSLO `schedule_sslo()`의 `scheduler.py:2463-2466`)가 onload가 타는 경로다. `has_mamba_layers`는 KV cache group 중 `MambaSpec` 존재 여부(`kv_cache_interface.py:768-771`)이고, `MambaSpec`은 Mamba1/2·GDN·linear-attn·ShortConv 레이어 전부가 만든다.
- **G1' — offload 구성에서는 mamba 모델이 자동으로 `align`이 된다.**
  커넥터가 prefix caching을 전제(G4)하고, prefix caching ON에서 `SupportsMambaPrefixCaching` 미지원 모델은 `align`으로 강제된다 (`models/config.py:338-406`). 이 프로토콜을 선언한 모델은 이 체크아웃에서 **Zamba2 하나뿐**.
- **G2 — 커넥터는 full-attention 그룹 1개 이상 필요.**
  `simple_kv_offload/manager.py:91-97` assert → 순수 Mamba(FalconMamba 등)는 엔진 기동 자체가 실패. 또한 DCP/PCP=1 필수(`manager.py:109`), CPU 코디네이터가 `use_eagle=False` 고정이라 eagle/MTP spec decode와 병용 금지.
- **G3 — HMA(hybrid KV cache manager) 딜레마.**
  `kv_transfer_config`가 있으면 기본값이 HMA OFF(`config/vllm.py:1292-1319`)인데, HMA OFF + mamba hybrid는 spec 통일 실패로 기동 불가(`kv_cache_utils.py:1399-1404`). HMA ON이면 G1 assert. **mamba hybrid는 양쪽 다 막힘.** SWA hybrid는 HMA OFF 시 full로 승격되어 동작(메모리 비효율만). `run_test.py`는 `disable_hybrid_kv_cache_manager=False`(HMA ON, 커넥터가 `SupportsHMA`라 정당).
- **G4 — prefix caching = offload 전제.** `simple_cpu_offload_connector.py:82-87` (없으면 조용히 비활성 → SSLO 쪽에서 ValueError).
- **G5 — encoder-decoder(CrossAttentionSpec) 비호환.** external tokens 전파 시 `CrossAttentionManager`가 caching 미지원.
- 참고: `MLAAttentionSpec`은 `FullAttentionSpec`의 서브클래스라 구조적으로는 통과.

## 3분류

**호환 (순수 full-attention, dense/MoE — 단일 FullAttentionSpec 그룹):**
Llama 3.x (in-repo `tests/v1/simple_kv_offload/test_integration.py`에서 실제 검증), Qwen2 dense, Qwen3 dense, **Qwen3-MoE (`Qwen3MoeForCausalLM`)**, Qwen3-VL 텍스트 디코더, Mistral/Mixtral, MiniMax-M2 등.

**비호환 (mamba/linear-attn hybrid — 전부 MambaSpec 그룹 생성):**
Qwen3.5/Qwen3.5-MoE(GDN; **크래시 실증됨**), Qwen3-Next(+MTP), Jamba, Bamba, FalconH1, NemotronH, Granite 4(GraniteMoeHybrid), MiniMax-Text-01/M1, LFM2 계열(ShortConv), Kimi-Linear, BailingMoe-V2.5, Olmo-Hybrid, PLaMo-2. 순수 Mamba/Mamba2/FalconMamba는 G2로 기동 실패. (Zamba2만 `mamba_cache_mode="all"`로 G1을 우회하지만 external-token 경로 미검증.)

**조건부/불확실:**
- Gemma-3 텍스트(5:1 SWA hybrid): in-repo 테스트 있음 — 사용 가능하되 onload 후 출력 동등성 검사 권장.
- gpt-oss(SWA+sink): perf 테스트에만 등재.
- Gemma-4(SWA hybrid + KV 공유 + 이종 head_dim): assert는 없지만 커넥터의 그룹 대칭 가정과 충돌 소지 — 미검증.
- DeepSeek MLA(V2/V3): 구조적 호환, 압축 경로 미검증. V4(그룹별 상이 block_size)는 비권장.
- spec decode(eagle/MTP), DCP/PCP>1, encoder-decoder: offload와 병용 금지.

## 커넥터 교체(LMCache 등)로는 해결되지 않음

G1 assert는 커넥터 공통 경로에 있다: waiting 루프가 v1 `KVConnectorBase` 인터페이스의 `get_num_new_matched_tokens()`를 호출하고(`scheduler.py:762-780`) 그 결과가 mamba-hybrid에서 `_mamba_block_aligned_split()`로 전달되므로(`scheduler.py:856-864`, `2463-2468`), LMCache/NIXL/Mooncake/`offloading_connector` 등 어떤 v1 커넥터를 써도 같은 지점에서 죽는다. LMCache는 paged KV 블록 저장 프레임워크라 recurrent인 mamba/linear-attn 상태를 복원할 의미론 자체가 없고(`lmcache_connector.py`에 mamba 처리 0건), G3(HMA 딜레마)도 `kv_transfer_config` 존재만으로 발동돼 커넥터 불문 동일하다. hybrid 지원은 커넥터 선택 문제가 아니라 mamba 상태 블록 직접 스왑 + align 장부 정합화라는 신규 구현 문제다.

## 실무 결론 (실험 모델 축)

1. **1순위**: full-attention MoE **Qwen3-30B-A3B**(로컬 `/cache/models/Qwen3-Coder-30B-A3B-Instruct` 보유) — 기존 35B-A3B의 "MoE 중간 규모" 성격 유지 + offload 완전 호환. dense 축은 Qwen3-8B/14B 또는 Llama-3.1-8B.
2. 2순위: Gemma-3 텍스트 (정확도 회귀 검사 전제).
3. 금지: mamba/linear-attn hybrid 전부.
4. 고정 구성: `enable_prefix_caching=True`, `disable_hybrid_kv_cache_manager=False`, DCP/PCP=1, spec decode 비활성.
5. Qwen3.5 축을 살리려면 assert 제거가 아니라 (a) `_mamba_block_aligned_split`의 external tokens 정렬 확장, (b) `MambaManager.allocate_new_computed_blocks` 오버라이드로 align 장부 정합화, (c) CPU 코디네이터의 mamba 상태 블록 back-trace 검증이 필요 — 일정상 비권장.
