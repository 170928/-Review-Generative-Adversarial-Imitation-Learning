# -Review-Generative-Adversarial-Imitation-Learning  
[Review &amp; Code]  
> Jonathan Ho, Stefano Ermon  
> https://arxiv.org/pdf/1606.03476.pdf  

> 참조 사이트   
> http://bboyseiok.com/GAIL/

## [Motivation]
전문가 (expert)와의 상호 작용이나 reward signal에 대한 정보 없이 (접근 없이) 전문가의 행동으로부터 정책 (policy) 을 학습하는 것을 고려하고자 하는 경우가 많습니다.    
이를 위해서 inverse reinforcement learning을 통해 전문가의 cost function를 복구 한 다음 reinforcement learning을 통해 해당 cost function에 맞는 정책 (policy)를 추출하는 방법이 사용되어 왔습니다.  
그러나 이러한 방법은 속도면에서 굉장히 비효율 적이었습니다.  

이와 같은 문제에 적합한 두 가지 주요 접근법이있습니다.  
1. 전문가의 행동으로부터 (state, action)에 대한 감독 학습 문제 (supervised learning problem)으로서 정책 (policy)을 습득하는 "behavior cloning"  
2. 전문가의 행동에서 cost function을 찾는 "inverse reinforcement learning"  

그 중 성공적인 결과를 이끌어내어 많은 사용을 받는  
Inverse Reinforcement Learning이 가지는 단점은 실행시에 필요한 연산 입니다.    
> expensive to run 이라고 표현됩니다.  
이는 전문가 (expert)의 행동에서 cost function을 학습하는 과정에서 reinforcement learning이 inner loop에서의 반복적인 수행이 필수적이기에 발생하게 됩니다.  

학습자 (learner) 의 목표 (objective)가 전문가를 모방하는 행동을 취하는 것을 감안할 때, 실제로 많은 IRL 알고리즘은 학습 한 비용의 최적 동작 (actions)의 품질에 대해 평가됩니다.  
그러나, IRL에서 cost function을 학습하는 것은 computational expense를 발생시키면서도 직접적으로 action을 만들어 내는 것에 실패합니다.    

그러므로!!  
이 논문에서는 정책을 직접 학습함으로써 행동하는 법 (policy)를 명시 적으로 알려주는 알고리즘을 제안합니다.  
최대 인과성 엔트로피 IRL (maximum causal entropy IRL)에 의해 학습 된 cost function에 대해 Reinforcement learning을 실행함으로써 주어진 정책을 특성화합니다. 그리고 이 특성화는 중간 IRL 단계를 거치지 않고 데이터에서 직접 정책을 학습하기위한 프레임 워크에 사용됩니다.

### [Behavior Cloning vs Inverse Reinforcement Learning]
(1) Behavior Cloning는 단순한 방법 이지만, covariate shift로 인한 compound error가 발생하여 성공적이 결과를 위해서는 많은 양의 데이터가 필요합니다.  
(2) Inverse Reinforcement Learning은 trajectories보다 전체 trajectories를 우선으로하는 cost function을 학습하므로 single-time step의 결정 (decision) 문제에 대해서 학습이 fit되는 것과 같은 오류가 문제가되지 않습니다.  

따라서 Inverse Reinforcement Learning은 택시 운전자의 행동 예측에서부터 네발로 된 로봇의 발판을 계획하는 데 이르기까지 광범위한 문제에 성공했습니다.

## [Background]
> Background는 기본적은 Reinforcement에서 적용되는 용어들을 사용하므로 다음과 같이 사진으로 대체합니다.  
![image](https://user-images.githubusercontent.com/40893452/46005029-4f99db00-c0ef-11e8-8c08-0e0400a1bde0.png)

> ![image](https://user-images.githubusercontent.com/40893452/46005291-e797c480-c0ef-11e8-812e-3840a726215e.png)



## [Related Work]

기본적으로 PPO, TRPO 의 알고리즘이 사용되므로 살펴보시는 것을 추천합니다. 
> 이원웅 님의 TRPO 에 대한 좋은 슬라이드 입니다.  
> https://www.slideshare.net/WoongwonLee/trpo-87165690

> PPO 논문입니다  
> https://arxiv.org/pdf/1707.06347.pdf  
> TRPO 논문입니다   
> https://arxiv.org/abs/1502.05477  

> 두 논문 모두 내용과 구현이 굉장히 어려운 논문이라... 천천히해보려고합니다.

## [Details]







