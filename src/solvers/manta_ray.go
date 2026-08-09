package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	mantaRayPopulation = 20
	mantaRayMaxSteps   = 1000
	mantaRayTolerance  = 1e-6
	mantaRayRangeLow   = -5.0
	mantaRayRangeHigh  = 5.0
)

// MantaRay minimizuje konfigurisanu benchmark funkciju koristeći Manta Ray Foraging Optimization: raja nasumično
// bira lančano hranjenje (ka najboljem rešenju i prethodnoj jedinki), ciklonsko hranjenje (spiralno ka najboljem
// opadajućom amplitudom) ili somersault preokret oko najboljeg rešenja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func MantaRay(problem models.Problem) (result models.Result, err error) {

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

	population := mantaRayPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := mantaRayMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := mantaRayTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	rays := make([][]float64, population)
	values := make([]float64, population)
	for i := range rays {
		ray := make([]float64, dimensions)
		for d := range ray {
			ray[d] = mantaRayRangeLow + rand.Float64()*(mantaRayRangeHigh-mantaRayRangeLow)
		}
		rays[i] = ray
		values[i] = fn.Evaluate(ray)
	}

	best := append([]float64(nil), rays[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), rays[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range rays {
			p := rand.Float64()
			switch {
			case p < 0.33:
				prev := rays[max(0, i-1)]
				for d := range rays[i] {
					rays[i][d] = rays[i][d] + rand.Float64()*(best[d]-rays[i][d]) + rand.Float64()*(prev[d]-rays[i][d])
				}
			case p < 0.66:
				// Originalna formula je u potpunosti zamenjivala poziciju (best + šum) bez ikakvog
				// inkrementalnog pomeraja od trenutne pozicije — bezuslovna resempl-oko-best pretraga
				// lako naiđe na duge platoe bez poboljšanja, pa noImprove logika prekine izvršavanje
				// davno pre optimuma. Zamenjeno pravim inkrementalnim pomerajem ka best-u uz opadajući šum
				for d := range rays[i] {
					rays[i][d] = rays[i][d] + rand.Float64()*(best[d]-rays[i][d]) + (rand.Float64()*2-1)*(1-float64(steps)/float64(maxSteps))*0.5
				}
			default:
				for d := range rays[i] {
					rays[i][d] = rays[i][d] + rand.Float64()*2*(best[d]-rays[i][d])
				}
			}

			for d := range rays[i] {
				rays[i][d] = clamp(rays[i][d], mantaRayRangeLow, mantaRayRangeHigh)
			}
			values[i] = fn.Evaluate(rays[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), rays[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
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
		Method:     "manta_ray",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
