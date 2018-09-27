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
> https://medium.com/@sanketgujar95/generative-adversarial-imitation-learning-266f45634e60   
> PPO 논문입니다  
> https://arxiv.org/pdf/1707.06347.pdf  
> TRPO 논문입니다   
> https://arxiv.org/abs/1502.05477  

> 두 논문 모두 내용과 구현이 굉장히 어려운 논문이라... 천천히해보려고합니다.

## [Details]




## [Basic Topics]
> 위 논문을 이해하기 위해 공부하게 된 개념들에 대한 정리부분 입니다.  
> 논문과는 무관합니다.  

### 1st Order TRPO 의 Kullback-Leibler Divergence     
> https://blog.naver.com/atelierjpro/220981354861  

[1] KL divergence term을 통해서 neural network의 학습 과정에서 network가 크게 변하는 것을 방지하기 위한 목적으로 사용되는 penalty term 입니다.    
[2] 랜덤하게 발생하는 event가 x, event x가 발생할 확률을 p(x) 라 가정합니다.  
이때, 낮은 확률의 정보를 얻게 될 수록 해당 event를 관찰하는 사람은 "크게 놀라게" 됩니다.   
고속도로에서 국산 차가 지나가면 놀라지 않지만, 가끔 고급 페라리와 같은 외제차가 지나가게 되면 놀라게 되는 예시가 가능합니다.  
x = 페라리 p(페라리) = 페라리 일 확률  
x = 아반떼 p(아반떼) = 아반떼 일 확률  
p(페라리) <<<< p(아반떼)  
위의 가정에서 x와 p(x) 의 관계를 log를 이용해서 다음과 같이 표현할 수 있습니다.   
정보의 양 : h(x) = -log(p(x))  
log 함수의 성질에 의해서 p(x) => 0 에 가까워 지면 무한대로 h(x)가 증가하게 됩니다.  
반면, p(x) = 1 이면 항상 일어나는 일로써 볼 수 있고, h(x) = 0 이 됩니다. 즉, 새로운 정보의 양이 0 이라는 의미로 볼 수 있습니다.  

[3] 엔트로피  
여러가지 사건들이 발생하는 경우, 엔트로피는 위에서 정의한 h(x)의 weighted sum으로써 정의 됩니다.  
즉, 페라리가 지나가고 아반떼가 지나 가는 것과 같은 경우를 예시로 볼 수 있습니다.  
p(페라리) = 0.1  p(아반떼) = 0.9   
Entropy = -(0.1)*log(0.1) - (0.9)*log(0.9) 가 됩니다.    
이때 주의할 점은, 일어난 사건들이 가지는 확률의 합이 1 이 되어야 합니다.  
Entropy = - Sigma_x p(x)log(p(x)) 일때, p(x)의 합이 1이라는 것을 의미합니다.  

[4] KL Divergence  
위의 개념들을 합하여 KL Divergence를 이해할 수 있게 됩니다.  
정확한 형태를 모르는 확률 분포 p(x)가 있다고 가정합니다.  
위와 유사한 예를 들어 보면 다음과 같습니다.  
p(페라리) = 0.01, p(아반떼) = 0.8, p(etc1) = ... , p(etc2) = ..., .....  
이때, 모든 event 발생의 확률의 합은 1 이 되어야 한다는 가정이 존재합니다.  
그러나 우리는 각 이벤트가 발생할 확률에 대한 정확한 지식이 없습니다.  
그러므로, 추정을 하게 되고 추정된 확률을 q(x)라고 합니다.  
이 q(x)는 실제 분포 p(x)와는 다릅니다.  
그 결과 h(x)로써 위에서 언급했던 정보의 양이 추측된 분포 q(x)에서와 실 분포 p(x)에서 다르게 되며,  
이 다른 정보의 양의 차이를 KL Divergence라고 합니다.   
![image](https://user-images.githubusercontent.com/40893452/46066756-2ee58a00-c1b0-11e8-8a68-38982c216d93.png)







