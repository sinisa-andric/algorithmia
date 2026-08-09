package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	quickpropLearningRate = 0.1
	quickpropEpsilon      = 1e-6
	quickpropMaxGrowth    = 1.75
	quickpropMaxSteps     = 1000
	quickpropTolerance    = 1e-6
	quickpropStepClip     = 1.0
	quickpropInitStep     = 0.01
)

// Quickprop minimizuje konfigurisanu benchmark funkciju koristeći Quickprop: klasična sekantna heuristika koja
// procenjuje kvadratni minimum iz promene gradijenta između dva uzastopna koraka, sa eksplicitnim ograničenjem
// rasta koraka — potpuno drugačiji princip od svih EMA/momentum pristupa u ovom modulu
// problem.Point je početna tačka pretrage
func Quickprop(problem models.Problem) (result models.Result, err error) {

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

	learningRate := quickpropLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	epsilon := quickpropEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxGrowth := quickpropMaxGrowth
	if v, ok := problem.Payload["max_growth"].(float64); ok {
		maxGrowth = v
	}

	maxSteps := quickpropMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := quickpropTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	gradPrev := make([]float64, dimensions)
	stepPrev := make([]float64, dimensions)
	for d := range stepPrev {
		stepPrev[d] = quickpropInitStep
	}

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		delta := make([]float64, dimensions)
		for d := range delta {
			denom := gradPrev[d] - grad[d]
			var deltaD float64
			// na prvom koraku gradPrev/stepPrev su izmišljeni placeholderi (nema stvarne prethodne
			// putanje), pa bi sekantna formula računala na fiktivnim podacima i gurala x u POGREŠNOM
			// pravcu. Prvi korak zato uvek koristi običan gradijentni korak
			if steps > 0 && math.Abs(denom) > epsilon {
				deltaD = grad[d] * stepPrev[d] / denom
			} else {
				deltaD = learningRate * grad[d]
			}

			growthBound := maxGrowth * math.Abs(stepPrev[d])
			if growthBound < 1e-3 {
				growthBound = 1e-3
			}
			deltaD = clamp(deltaD, -growthBound, growthBound)

			delta[d] = -deltaD
		}
		delta = clipStep(delta, quickpropStepClip)
		for d := range x {
			x[d] += delta[d]
			// stepPrev mora čuvati ISTU konvenciju predznaka koja ulazi u sekantnu formulu iduće iteracije
			// (grad*step_prev/(grad_prev-grad)), tj. -delta (post-clip primenjeni pomeraj sa ispravljenim
			// predznakom) — čuvanje samog delta (već negiranog radi x+=delta primene) obrtalo je predznak
			// sekantne procene svakog narednog koraka i teralo x da monotono divergira do granice clip-a
			// umesto da se približava optimumu
			stepPrev[d] = -delta[d]
		}
		gradPrev = grad

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
		Method:     "quickprop",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
