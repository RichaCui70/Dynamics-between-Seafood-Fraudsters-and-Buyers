INTRO = r'''
    # Modeling the Time Dynamics of Seafood Harvesting: Interactions between Buyers, Fraudsters, and Resource Sustainability

    ## Introduction
    Seafood is one of the most heavily traded food comodities in the era of globalization (Shehata et al. 2016). Being around a quarter of the global intake of animal protein, seafood has a huge economy, with around one-tenth of the world's population living off revenue generated from fisheries. With such a demand for seafood in the modern era, the pressure to meet the global demand, taken together with unethical fishing tactics and the complexities of global supply chains make it increasingly importantly, yet difficult to ensure the authenticity and traceability of seafood products on the market (Naaum et al. 2016). In other words, it's diffucult to track where seafood fraud can occur in the supply chain. Seafood fraud can be defined as the act of mislabelling, substituting, adulterating, and/or unethically harvesting seafood.

    Keeping in mind the difficulty, this presents the opportunity to model the degree of interactions existing among key players in the supply chain through the modeling of dynamical systems.

    ## The model in 4 parts
    With the proposed system, it is segmented into 4 bins: Seafood, Fishers, Wholesalers, and Buyers

    ### Seafood harvesting

    The relationship between fishers and seafood is shown through the effort $(E)$ of fishers in harvesting seafood $(S)$. Seafood has an intrinsic growth rate $(r)$ and some intrinsic carrying capacity $(K)$. Fishers' effort grows when the cost of fishing $(C_{t})$ decreases, and the revenue of fishing $(qP^{w}_{t}S_{t})$ increases, where $q$ represents seafood catchability and $P^{w}_{t}$ represents the price per unit of seafood. In other words, effort grows when their profit margin expands. As effort grows, they harvest more seafood ($qE_{t}$), regualting/decreasing seafood levels, vice versa. The relationship between effort and seafood is shown through:
    
    $
    S_{t+1} = S_{t}e^{\gamma_{S}(r(1-\frac{S_{t}}{K}) - q_{t}E_{t})} \\
    E_{t+1} = E_{t}e^{\gamma_{E}(q_{t}P^{w}_{t}S_{t} - C_{t})}
    $
    
    In this system, $K$ is always set to one, such that bounding seafood between zero and one allows for a more simple mathematical analysis of the system. It's easier define as well: seafood at one representing natural maximum amount of seafood the environment can carry, and seafood at zero representing extinction.

    Effort is shown as any non-negative number. This tells us the magnitude of their efforts in harvesting seafood. Their effort directly correlates to the amount of seafood harvested $(H)$, shown through:
    
    $
    q_{t} = (q_{1}-q_{0})F_{t} + q_{0}\\
    H_{t} = q_{t}E_{t}S_{t}
    $
    
    Harvest can congruent to the seafood supply available to wholesalers and buyers.

    ### Fishers v. Wholesalers

    Fishers and wholesalers have a dynamic, such that they compete to gain the best profit margin. Fishers look to minimize their costs of fishing and to maximize their revenue, whereas wholesalers look to maximize their market prices for seafood consumers to pay, while minimizing the wholesale prices that they buy the fish for.

    Now enter fraudsters as a proportion of wholesalers $(F)$. Fraudsters enable fishers to partake in illegal fishing methods such that they can gain an abundance of seafood at a much lower cost. Fraudsters can then negotiate a lower wholesale price due the illegal nature of the harvest and take the risk of purchasing illegally fished seafood. This way, both parties can have participants looking to increase their profit margins.

    It's important to note that the fraudsters in the wholesale enable this fraud, such that without them, there is no market for this illegally caught seafood, therefore giving no motivation for fishers to use illegal harvesting methods. This gives us way to the following wholesale price and fishing cost equations:

    $
    C_{t} = (C_{1}-C_{0})F_{t} + C_{0}\\
    P^{w}_{t} = \frac{(P^{w}_{1}-P^{w}_{0})F_{t} + P^{w}_{0}}{(\gamma_{p}H_{t})^{ϵ_{s, w}}}
    $
    
    Where the subscipt of one represents the cost of fishing and/or wholesale price when 100 percent of wholesalers are fraudsters, and zero represents the cost of fishing and/or wholesale price when zero percent of wholesalers are fraudsters. $ϵ_{s, w}$ represents the supply elasticity - how sensitive the wholesale price of fish is to the given supply.

    Because fraudsters are looking to maximize their profit margins, the fraudster equation is given as such:
    
    $
    F_{t+1} = \frac{F_{t}e^{\gamma_{F}(P^{m}_{t} - P^{w}_{t})}}{1+F_{t}(e^{\gamma_{F}(P^{m}_{t} - P^{w}_{t}})-1)}
    $
    
    The fraudsters equation follows a logistic-like function that bounds $F_{t}$ between zero and one. This allows the fraudsters to be represented as a proportion of the total amount of wholesalers within the seafood supply chain. $F_{t}$ equaling one means that every wholesaler is fraudulent, and $F_{t}$ equaling zero means that every wholesaler is honest.
    ### Wholesalers v. Buyers

    Focusing now on buyers, they don't want to be taken advantage of. They would like whatever they buy to be ethically sourced and to be labelled accurately. Thus, they have an awareness of fraudsters in the supply chain. Whether through the news, or figuring it out themselves, the more fraudsters there are in the system, the higher chance that buyers will figure out that there's foul play involved in the seafood. The higher their perception of fraud in the markets $(F^{p}_{t})$, the lower their demand $(D_{t})$ will be, and the lower the market prices of seafood $(P^{m}_{t})$ will become.

    Thus, we are introduced to these equations:
    
    $
    F^{P}_{t+1} = \frac{F^{p}_{t}e^{\gamma_{FP}(F_{t} - \hat{F})}}{1+F^{p}_{t}(e^{\gamma_{FP}(F_{t} - \hat{F}})-1)}\\
    D_{t} = \frac{(1-F^{p}_{t})^{ϵ_{d}}}{P^{m}_{t}} = \sqrt{(1-F^{p}_{t})^{ϵ_{d}}H_{t}^{ϵ_{s, m}}}\\
    P^{m}_{t} = \gamma_{M}\frac{D_{t}}{H_{t}^{ϵ_{s, m}}} = \gamma_{M}\sqrt{\frac{(1-F^{p}_{t})^{ϵ_{d}}}{H_{t}^{ϵ_{s, m}}}}
    $
    
    The perception of fraudster equation follows the same functional form as the fraudster equation, such that it's bounded between zero and one to represent a proportion of the entire population of seafood buyers within the supply chain. $F^{p}_{t}$ at one means that every buyer believes there is some sort of fraudulent activity within the seafood supply chain, and $F^{p}_{t}$ at zero means that every buyer believes that every player within the seafood supply chain is honest (non-fraudulent). The way $F^{p}_{t}$ grows is through the actual amount of fraud $F_{t}$ growing bigger than some minimal noticeable level of fraud ($\hat{F}$). Therefore, when $F_{t}$ grows beyond $\hat{F}$, $F^{p}_{t}$ grows because they are able to pick up on the fraudulent activity present. Conversely, when $F_{t}$ shrinks below $\hat{F}$, $F^{p}_{t}$ shrinks because there isn't any noticeable fraudulent activity.

    Furthermore, as mentioned earlier, $D_{t}$ shrinks as $F^{p}_{t}$ increases. $D_{t}$'s sensitivity to fluctuating $F^{p}_{t}$ is given by $ϵ_{d}$, noted as the elasticity of demand against the perception of fraud. At a lower level of $ϵ_{d}$, fluctuating $F^{p}_{t}$ levels doesn't change demand much. At a higher level of $ϵ_{d}$, slight fluctuations in $F^{p}_{t}$ levels changes demand greatly. Because demand is also based off of the market price, it also takes into account the actual amount of supply available to purchase. Similar to $ϵ_{s, w}$ mentioned earlier, $ϵ_{s,m}$ represents the market prices' sensitivity to fluctuating seafood supply/harvest ($H_{t}$) levels. At a low level, prices don't fluctuate as much. At a high level, prices fluctuate greatly as $H_{t}$ changes.
'''

SEAFOOD_EFFORT = r'''
    $
    S_{t+1} = S_{t}e^{\gamma_{S}(r(1-\frac{S_{t}}{K}) - qE_{t})} \\
    E_{t+1} = E_{t}e^{\gamma_{E}(qP^{w}_{t}S_{t} - C_{t})}
    $
    
    
'''