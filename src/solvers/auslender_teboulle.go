package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	auslenderTeboulleMaxSteps     = 1000
	auslenderTeboulleLearningRate = 0.1
	auslenderTeboulleLambda       = 0.01
	auslenderTeboulleTolerance    = 1e-6
	auslenderTeboulleStepClip     = 1.0
)

// AuslenderTeboulle minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći Auslender-Teboulle metod:
// održava dve sekvence — brzu x (trenutna tačka) i z (proksimalno-gradijentna sekvenca ažurirana u tački y), za
// razliku od fista.go/optimal_gradient_method.go čiji je momentum multiplikativna korekcija, ovde se x i z
// kombinuju opadajućim tau po ekstragradijent šemi
// problem.Point je početna tačka pretrage
func AuslenderTeboulle(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := auslenderTeboulleMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := auslenderTeboulleLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := auslenderTeboulleLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := auslenderTeboulleTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x0 := append([]float64(nil), problem.Point...)
	x := append([]float64(nil), x0...)
	z := append([]float64(nil), x0...)

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		tau := 2.0 / (float64(steps) + 2)
		y := make([]float64, dimensions)
		for d := range y {
			y[d] = (1-tau)*x[d] + tau*z[d]
		}

		// izvorna formula akumulira k*grad u rastuću sumu i deli je fiksnim learning_rate-om — pošto suma
		// raste ~O(k^2) dok korak ostaje fiksan, efektivni pomeraj z-a je rastao bez ograničenja umesto da se
		// smanjuje, što je izazivalo nestabilnu oscilaciju (izmereno: z je oscilovao do ±21 na sphere, gde je
		// tačan optimum 0). Zamenjeno standardnim proksimalno-gradijentnim korakom na z korišćenjem TRENUTNOG
		// gradijenta u y (ne akumulirane sume) — matematički legitimna varijanta dual-averaging šeme koja
		// stabilno konvergira
		grad := numGrad(fn.Evaluate, y, proximalGradEps)

		zRaw := make([]float64, dimensions)
		for d := range zRaw {
			zRaw[d] = z[d] - learningRate*grad[d]
		}
		zCandidate := proxL1(zRaw, learningRate*lambda)

		zDelta := make([]float64, dimensions)
		for d := range zDelta {
			zDelta[d] = zCandidate[d] - z[d]
		}
		zDelta = clipStep(zDelta, auslenderTeboulleStepClip)
		for d := range z {
			z[d] += zDelta[d]
		}

		xCandidate := make([]float64, dimensions)
		for d := range xCandidate {
			xCandidate[d] = (1-tau)*x[d] + tau*z[d]
		}
		xDelta := make([]float64, dimensions)
		for d := range xDelta {
			xDelta[d] = xCandidate[d] - x[d]
		}
		xDelta = clipStep(xDelta, auslenderTeboulleStepClip)
		for d := range x {
			x[d] += xDelta[d]
		}

		value := objective(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), false)
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
		Method:     "auslender_teboulle",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
