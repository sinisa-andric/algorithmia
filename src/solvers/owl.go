package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	owlPopulation = 20
	owlMaxSteps   = 1000
	owlTolerance  = 1e-6
	owlRangeLow   = -5.0
	owlRangeHigh  = 5.0
)

// Owl minimizuje konfigurisanu benchmark funkciju koristeći Owl Search Algorithm: svaka sova prati intenzitet zvuka
// (obrnuto proporcionalan vrednosti funkcije) i kreće se ka najboljem rešenju srazmerno odnosu sopstvenog i
// najboljeg intenziteta, uz nasumičnu perturbaciju za ostatak
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Owl(problem models.Problem) (result models.Result, err error) {

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

	population := owlPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := owlMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := owlTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	owls := make([][]float64, population)
	values := make([]float64, population)
	for i := range owls {
		owl := make([]float64, dimensions)
		for d := range owl {
			owl[d] = owlRangeLow + rand.Float64()*(owlRangeHigh-owlRangeLow)
		}
		owls[i] = owl
		values[i] = fn.Evaluate(owl)
	}

	best := append([]float64(nil), owls[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), owls[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		bestIntensity := 1 / (bestValue + 1e-10)
		// Sova daleko od best-a ima nizak intenzitet (prob blizu 0), pa je originalna formula gotovo
		// u potpunosti prepuštala takve jedinke fiksnoj (ne-opadajućoj) šum-amplitudi bez privlačenja
		// ka best-u — dodata je bazna privlačnost ka best-u za sve sove i opadajuća šum-amplituda
		frac := 1 - float64(steps)/float64(maxSteps)

		for i := range owls {
			intensity := 1 / (values[i] + 1e-10)
			prob := intensity / (bestIntensity + 1e-10)

			for d := range owls[i] {
				owls[i][d] = owls[i][d] + (0.5+prob*0.5)*rand.Float64()*(best[d]-owls[i][d]) + (1-prob)*(rand.Float64()*2-1)*0.5*frac
				owls[i][d] = clamp(owls[i][d], owlRangeLow, owlRangeHigh)
			}
			values[i] = fn.Evaluate(owls[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), owls[i]...)
			}
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
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "owl",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
