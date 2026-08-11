- 추가 좁힘(정적 분석): `blocks_needed = (k*_unconstrained − k*) × per_admit`이
  결함 1의 per_admit=8을 재사용 → 두 결함은 같은 상수를 공유(독립 아님).
  offload 게이트 4종 중 min_residency(기본 0)는 배제. 선행 가설: 
  `is_fully_mirrored`가 in-flight store 존재 시 False(manager.py:676)라
  디코딩 중에는 미러링이 계속 밀리고, deferred 체류가 짧으면 영구 자격 미달.
  게이트별 기각 카운터 계측 필요 (미착수).
