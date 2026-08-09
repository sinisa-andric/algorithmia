package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"sort"
)

const (
	nelderMeadRestartMaxSteps    = 1000
	nelderMeadRestartRestarts    = 5
	nelderMeadRestartTolerance   = 1e-6
	nelderMeadRestartInitialStep = 0.1
	nelderMeadRestartPerturb     = 0.5
)

// NelderMeadRestart minimizuje konfigurisanu benchmark funkciju koristeći ponavljani Nelder-Mead: svaki restart
// kreće od male perturbacije najbolje dosad pronađene tačke
// problem.Point je početna tačka prvog pokretanja Nelder-Mead-a
func NelderMeadRestart(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := nelderMeadRestartMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	restarts := nelderMeadRestartRestarts
	if v, ok := problem.Payload["restarts"].(float64); ok {
		restarts = int(v)
	}
	if restarts < 1 {
		restarts = 1
	}

	tolerance := nelderMeadRestartTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	stepsPerRestart := maxSteps / restarts
	if stepsPerRestart < 1 {
		stepsPerRestart = 1
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	globalBest := append([]float64(nil), problem.Point...)
	globalBestValue := fn.Evaluate(globalBest)

	totalSteps := 0
	for r := 0; r < restarts; r++ {

		var start []float64
		if r == 0 {
			start = append([]float64(nil), problem.Point...)
		} else {
			start = make([]float64, len(problem.Point))
			for d := range start {
				start[d] = globalBest[d] + nelderMeadRestartPerturb*randNorm()
			}
		}

		point, value, steps := nelderMeadRun(fn, start, stepsPerRestart, tolerance)
		totalSteps += steps

		if value < globalBestValue {
			globalBestValue = value
			globalBest = point
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, totalSteps, globalBest, globalBestValue, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, totalSteps, globalBest, globalBestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "nelder_mead_restart",
		Point:      globalBest,
		Value:      globalBestValue,
		Steps:      totalSteps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

// nelderMeadRun izvršava standardni Nelder-Mead simpleks metod od start za
// najviše maxSteps iteracija i vraća najbolju pronađenu tačku, njenu
// vrednost i broj iteracija koje su bile potrebne.
func nelderMeadRun(fn functions.BenchmarkFunction, start []float64, maxSteps int, tolerance float64) ([]float64, float64, int) {

	dimensions := len(start)
	simplex := make([][]float64, dimensions+1)
	simplex[0] = append([]float64(nil), start...)
	for i := 0; i < dimensions; i++ {
		vertex := append([]float64(nil), start...)
		vertex[i] += nelderMeadRestartInitialStep
		simplex[i+1] = vertex
	}

	steps := 0
	for ; steps < maxSteps; steps++ {

		sort.Slice(simplex, func(i, j int) bool {
			return fn.Evaluate(simplex[i]) < fn.Evaluate(simplex[j])
		})

		best := fn.Evaluate(simplex[0])
		worst := fn.Evaluate(simplex[dimensions])
		if math.Abs(worst-best) < tolerance {
			break
		}

		centroid := make([]float64, dimensions)
		for _, vertex := range simplex[:dimensions] {
			for i, v := range vertex {
				centroid[i] += v / float64(dimensions)
			}
		}

		reflected := make([]float64, dimensions)
		for i := range centroid {
			reflected[i] = centroid[i] + (centroid[i] - simplex[dimensions][i])
		}
		reflectedValue := fn.Evaluate(reflected)

		secondWorst := fn.Evaluate(simplex[dimensions-1])

		switch {
		case reflectedValue < best:
			expanded := make([]float64, dimensions)
			for i := range centroid {
				expanded[i] = centroid[i] + 2*(reflected[i]-centroid[i])
			}
			if fn.Evaluate(expanded) < reflectedValue {
				simplex[dimensions] = expanded
			} else {
				simplex[dimensions] = reflected
			}

		case reflectedValue < secondWorst:
			simplex[dimensions] = reflected

		default:
			contracted := make([]float64, dimensions)
			for i := range centroid {
				contracted[i] = centroid[i] + 0.5*(simplex[dimensions][i]-centroid[i])
			}
			if fn.Evaluate(contracted) < worst {
				simplex[dimensions] = contracted
			} else {
				for v := 1; v <= dimensions; v++ {
					for i := range simplex[v] {
						simplex[v][i] = simplex[0][i] + 0.5*(simplex[v][i]-simplex[0][i])
					}
				}
			}
		}
	}

	sort.Slice(simplex, func(i, j int) bool {
		return fn.Evaluate(simplex[i]) < fn.Evaluate(simplex[j])
	})

	return simplex[0], fn.Evaluate(simplex[0]), steps
}
