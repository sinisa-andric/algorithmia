package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	prodigyLearningRate = 0.1
	prodigyBeta1        = 0.9
	prodigyBeta2        = 0.999
	prodigyEpsilon      = 1e-8
	prodigyMaxSteps     = 1000
	prodigyTolerance    = 1e-6
	prodigyStepClip     = 1.0
	// prodigyDInit=1e-6 (teorijski podrazumevani "hladan start" iz literature) je za ovaj budžet od 1000
	// koraka izazivao samopojačavajuće zamrzavanje: mala d -> mali pomeraj -> mala x-x0 -> mala dCandidate ->
	// d ostaje mala zauvek (izmereno: x ostaje zaglavljen na inicijalnoj tački, vrednost identična početnoj
	// 125/20 na sphere posle 15+ koraka). Veći početni d daje smislen prvi korak od kog se procena realno uči
	prodigyDInit = 1.0
)

// Prodigy minimizuje konfigurisanu benchmark funkciju koristeći Prodigy optimizator: skalar d koji množi efektivni
// learning_rate se adaptivno UČI iz istorije (procena udaljenosti od početne tačke projektovane na akumulirani
// gradijent), za razliku od bilo kog drugog solvera ovde gde je learning_rate fiksan ili zavisi samo od trenutnog
// koraka, ne od cele istorije
// problem.Point je početna tačka pretrage
func Prodigy(problem models.Problem) (result models.Result, err error) {

	if len(problem.Point) == 0 {
		err = fmt.Errorf("starting point is required")
		return result, err
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	learningRate := prodigyLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := prodigyBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := prodigyBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := prodigyEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := prodigyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := prodigyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x0 := append([]float64(nil), problem.Point...)
	x := append([]float64(nil), x0...)
	m := make([]float64, dimensions)
	s := make([]float64, dimensions)
	gradSum := make([]float64, dimensions)
	dEstimate := prodigyDInit

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		for d := range gradSum {
			gradSum[d] += grad[d]
		}

		dotProd := 0.0
		for d := range grad {
			dotProd += grad[d] * (x[d] - x0[d])
		}
		dCandidate := math.Abs(dotProd) / (norm(gradSum) + epsilon)
		dEstimate = math.Max(dEstimate, dCandidate)

		delta := make([]float64, dimensions)
		for d := range delta {
			m[d] = beta1*m[d] + (1-beta1)*grad[d]
			s[d] = beta2*s[d] + (1-beta2)*grad[d]*grad[d]
			delta[d] = -dEstimate * learningRate * m[d] / (math.Sqrt(s[d]) + epsilon)
		}
		delta = clipStep(delta, prodigyStepClip)
		for d := range x {
			x[d] += delta[d]
		}

		value := fn.Evaluate(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}
		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "prodigy",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
