# V1 Mean, Variance, and Probability

**mean $m$** is the average value or expected value

**variance $\sigma^2$** measures the average *square distance* from the mean $m$

**Probabilities** of $n$ different outcomes are positive numbers $p_1, p_2, \dots, p_n$ adding to 1



<font size=5><center>Sample Mean</center></font>




$$m = \mu =  \frac{1}{N}(x_1+x_2+\dots+x_N) $$






Also see Equation \@ref(eq:mean).
<font size=5><center>Expected Value</font></center>
The expected value of $x$ starts from its corresponding probabilities
$$
m = E \left[ x \right] = p_1x_1 + p_2x_2 + \dots + p_nx_n
$$
Note that $m = E\left[x\right]$ tells us waht we should expect while  $m=\mu$ tells us what we got.
<font size=5><center>Law of Large Numbers</font></center>
With probability 1, the sample mean $\mu$ will converge to its expected value $E\left[x\right]$ as the sample size $N$ increases

# Variance (around the mean)
The **Variance** $\sigma^2$ measures expected distance (squared) from the expected mean $E\left[x\right]$

The sample variance $S^2$ measures actual distance (squared) from the actual sample mean. The square root is the standard deviation $\sigma$ or $S$

<font size=5><center>Sample Variance</center></font>
$$S^2 = \frac{1}{N-1}\left[(x_1-m)^2 + \dots + (x_N - m)^2\right]$$

<font color="red"> Please notice! Statisticians divide by $N - 1 = 4$ (and not N = 5) so that $S^2$ is an unbiased estimate of a $\sigma^2$ One degree of freedom is already accounted for in the sample mean.</font>
#### Split the equation
$$ S^2 = \frac{1}{N-1}\left[\sum_{i=1}^{n} (x_i - m)^2 \right]$$
$$ S^2 = \frac{1}{N-1} \left[\sum_{i=1}^{n} (x_i)^2 - 2m\sum_{i=1}^{n}x_i + \sum_{i=1}^{n}m^2\right]$$
$$ S^2 = \frac{1}{N-1} \left[\sum_{i=1}^{n} (x_i)^2 - 2m(Nm) + Nm^2\right]$$
$$ S^2 = \frac{1}{N-1}\left[\sum_{i=1}^{n} x_i^2 - Nm^2\right]$$

<font size=5><center> Expected Variance<center></font>
$$\sigma^2 = E\left[(x-m)^2\right] = p_1(x_1-m)^2+\dots+p_n(x_n-m)^2$$

# Continous Probability Distributions

$p(x)$ is a probability distributation for some whole continuous range of variable $x$

$F(x)$ is the chance that a random variable is less that $x$, it is called the **cumulative distribution**

$F(x)$ is the **cumulative distribution** and its derivative $p(x) = dF/dx$ is the **probability density function (pdf)**

You could say that $p(x)dx$ is the probability of a sample falling in between $x$ and $x+dx$ and this is only "infinitesimally true" because $p(x)dx$ is $F(x+dx) - F(x)$ only works for dx that is infinitesimally small

Probability of $p(a \le x \le b) = \int_{a}^{b}p(x)dx=F(b)-F(a)$

## Mean and Variance of $p(x)$ Continous variable

<font size=5><center>Mean for a uniform distributation range from 17 to 20</center></font>
$$m = E\left[x\right] = \int_xp(x)dx = \int_{17}^{20}(x)(\frac{1}{3})dx = 18.5 $$

## Variance for Continous variable
$$ \sigma^2 = E\left[(x-m)^2\right]=\int p(x)(x-m)^2 dx $$、

# Normal Distribution

Gaussian desity $p(x)$

Normal distrbution $N(m, \sigma)$

$$p(x)=N(m, \sigma)=\frac{1}{\sigma\sqrt{2\pi}} e^{-(x-m)^2 / 2\sigma^2}$$