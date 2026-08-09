package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	tlboPopulation = 20
	tlboMaxSteps   = 500
	tlboTolerance  = 1e-6
	tlboRangeLow   = -5.0
	tlboRangeHigh  = 5.0
)

// Tlbo minimizuje konfigurisanu benchmark funkciju koristeći Teaching-Learning-Based Optimization: u fazi učitelja
// svaki student se pomera ka razlici između best-a i pojačanog proseka razreda, a u fazi učenika uči od nasumičnog
// boljeg kolege ili se udaljava od lošijeg, uz pohlepno prihvatanje samo poboljšanja u obe faze
// problem.Point inicijalizuje populaciju studenata i određuje njenu dimenzionalnost
func Tlbo(problem models.Problem) (result models.Result, err error) {

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

	population := tlboPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := tlboMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := tlboTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = tlboRangeLow + rand.Float64()*(tlboRangeHigh-tlboRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
	}

	best := append([]float64(nil), individuals[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), individuals[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		mean := make([]float64, dimensions)
		for _, individual := range individuals {
			for d := range mean {
				mean[d] += individual[d]
			}
		}
		for d := range mean {
			mean[d] /= float64(len(individuals))
		}

		// faza učitelja: teaching factor 1 ili 2 pojačava/slabi uticaj proseka razreda na pomeraj ka best-u
		for i := range individuals {
			tf := 1.0
			if rand.Float64() < 0.5 {
				tf = 2.0
			}
			candidate := make([]float64, dimensions)
			for d := range candidate {
				candidate[d] = clamp(individuals[i][d]+rand.Float64()*(best[d]-tf*mean[d]), tlboRangeLow, tlboRangeHigh)
			}
			candidateValue := fn.Evaluate(candidate)
			if candidateValue < values[i] {
				individuals[i] = candidate
				values[i] = candidateValue
			}
		}

		// faza učenika: uči od nasumičnog boljeg kolege (pomak ka njemu) ili se udaljava od lošijeg
		for i := range individuals {
			j := i
			if len(individuals) > 1 {
				j = rand.IntN(len(individuals))
				for j == i {
					j = rand.IntN(len(individuals))
				}
			}
			candidate := make([]float64, dimensions)
			if values[j] < values[i] {
				for d := range candidate {
					candidate[d] = clamp(individuals[i][d]+rand.Float64()*(individuals[j][d]-individuals[i][d]), tlboRangeLow, tlboRangeHigh)
				}
			} else {
				for d := range candidate {
					candidate[d] = clamp(individuals[i][d]+rand.Float64()*(individuals[i][d]-individuals[j][d]), tlboRangeLow, tlboRangeHigh)
				}
			}
			candidateValue := fn.Evaluate(candidate)
			if candidateValue < values[i] {
				individuals[i] = candidate
				values[i] = candidateValue
			}
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
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
		Method:     "tlbo",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
