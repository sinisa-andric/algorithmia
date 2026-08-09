package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	crystalStructurePopulation = 20
	crystalStructureMaxSteps   = 500
	crystalStructureTolerance  = 1e-6
	crystalStructureRangeLow   = -5.0
	crystalStructureRangeHigh  = 5.0
)

// CrystalStructure minimizuje konfigurisanu benchmark funkciju koristeći Crystal Structure Algorithm: svaki atom
// se svakog koraka nasumično kreće ka jednom od četiri tipa referentne tačke — best-u (osnovna ćelija), centru mase
// populacije (kubna simetrija), nasumičnom atomu uz vezu ka best-u, ili slobodno istražuje ka uglu rešetke
// problem.Point inicijalizuje populaciju atoma i određuje njenu dimenzionalnost
func CrystalStructure(problem models.Problem) (result models.Result, err error) {

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

	population := crystalStructurePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := crystalStructureMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := crystalStructureTolerance
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
			individual[d] = crystalStructureRangeLow + rand.Float64()*(crystalStructureRangeHigh-crystalStructureRangeLow)
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

		meanPoint := make([]float64, dimensions)
		for _, individual := range individuals {
			for d := range meanPoint {
				meanPoint[d] += individual[d]
			}
		}
		for d := range meanPoint {
			meanPoint[d] /= float64(len(individuals))
		}

		for i := range individuals {
			mode := rand.Float64()

			switch {
			case mode < 0.25:
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+rand.Float64()*(best[d]-individuals[i][d]), crystalStructureRangeLow, crystalStructureRangeHigh)
				}
			case mode < 0.5:
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+rand.Float64()*(meanPoint[d]-individuals[i][d])*0.5, crystalStructureRangeLow, crystalStructureRangeHigh)
				}
			case mode < 0.75:
				peerIdx := i
				if population > 1 {
					peerIdx = rand.IntN(population)
					for peerIdx == i {
						peerIdx = rand.IntN(population)
					}
				}
				for d := range individuals[i] {
					pullPeer := rand.Float64() * (individuals[peerIdx][d] - individuals[i][d]) * 0.5
					pullBest := rand.Float64() * (best[d] - individuals[i][d]) * 0.2
					individuals[i][d] = clamp(individuals[i][d]+pullPeer+pullBest, crystalStructureRangeLow, crystalStructureRangeHigh)
				}
			default:
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+(rand.Float64()*2-1)*0.8*(1-float64(steps)/float64(maxSteps)), crystalStructureRangeLow, crystalStructureRangeHigh)
				}
			}
			values[i] = fn.Evaluate(individuals[i])
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
		Method:     "crystal_structure",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
