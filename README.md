# -Review-Generative-Adversarial-Imitation-Learning  
[Review &amp; Code]  
> Jonathan Ho, Stefano Ermon  
> https://arxiv.org/pdf/1606.03476.pdf  

## [Motivation]
(1) 전문가 (expert)와의 상호 작용이나 reward signal에 대한 정보 없이 (접근 없이) 전문가의 행동으로부터 정책 (policy) 을 학습하는 것을 고려하고자 하는 경우가 많습니다.    
이를 위해서 inverse reinforcement learning을 통해 전문가의 cost function를 복구 한 다음 reinforcement learning을 통해 해당 cost function에 맞는 정책 (policy)를 추출하는 방법이 사용되어 왔습니다.  
그러나 이러한 방법은 속도면에서 굉장히 비효율 적이었습니다.  

이와 같은 문제에 적합한 두 가지 주요 접근법이있습니다.  
1. 전문가의 행동으로부터 (state, action)에 대한 감독 학습 문제 (supervised learning problem)으로서 정책 (policy)을 습득하는 "behavior cloning"  
2. 전문가의 행동에서 cost function을 찾는 "inverse reinforcement learning"  

(2) 


## [Methodology]
(1) Data 로 부터 직접 inverse reinforcement learning을 통해서 reinforcement learning으로 학습한 것과 같은 정책 (policy)를 추출해내는 framework를 제안  

(2) 

## [Related Work]


## [Details]

