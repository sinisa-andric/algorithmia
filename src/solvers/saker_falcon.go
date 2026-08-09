package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	sakerFalconPopulation = 20
	sakerFalconMaxSteps   = 1000
	sakerFalconTolerance  = 1e-6
	sakerFalconRangeLow   = -5.0
	sakerFalconRangeHigh  = 5.0
)

// SakerFalcon minimizuje konfigurisanu benchmark funkciju koristeći Saker Falcon Optimization: izvršavanje prolazi
// kroz tri faze leta — pretragu (širok nasumičan let), krstarenje (kretanje ka najboljem uz nasumičnost) i napad
// (brzo i precizno kretanje ka najboljem)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SakerFalcon(problem models.Problem) (result models.Result, err error) {

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

	population := sakerFalconPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := sakerFalconMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := sakerFalconTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	falcons := make([][]float64, population)
	values := make([]float64, population)
	for i := range falcons {
		falcon := make([]float64, dimensions)
		for d := range falcon {
			falcon[d] = sakerFalconRangeLow + rand.Float64()*(sakerFalconRangeHigh-sakerFalconRangeLow)
		}
		falcons[i] = falcon
		values[i] = fn.Evaluate(falcon)
	}

	best := append([]float64(nil), falcons[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), falcons[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		phase := float64(steps) / float64(maxSteps)

		for i := range falcons {
			switch {
			case phase < 0.33:
				// Čist nasumičan let bez ikakve veze sa best-om oslanja se isključivo na to da faza
				// pretrage potraje dovoljno dugo da naiđe na dobru tačku — ali noImprove logika često
				// prekine izvršavanje u ovoj fazi pre nego što uopšte dođe do krstarenja/napada, pa je
				// dodata puna privlačnost ka best-u (uz slabiji šum) da pretraga sistematski napreduje
				for d := range falcons[i] {
					falcons[i][d] = falcons[i][d] + (rand.Float64()*2-1)*0.3 + rand.Float64()*(best[d]-falcons[i][d])
				}
			case phase < 0.66:
				for d := range falcons[i] {
					falcons[i][d] = falcons[i][d] + rand.Float64()*(best[d]-falcons[i][d])
				}
			default:
				for d := range falcons[i] {
					falcons[i][d] = falcons[i][d] + 0.8*(best[d]-falcons[i][d])
				}
			}

			for d := range falcons[i] {
				falcons[i][d] = clamp(falcons[i][d], sakerFalconRangeLow, sakerFalconRangeHigh)
			}
			values[i] = fn.Evaluate(falcons[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), falcons[i]...)
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
		Method:     "saker_falcon",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
