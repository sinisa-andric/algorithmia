package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"sort"
)

const (
	nelderMeadAdaptiveMaxSteps    = 1000
	nelderMeadAdaptiveTolerance   = 1e-6
	nelderMeadAdaptiveInitialStep = 0.05
)

// NelderMeadAdaptive minimizuje sphere funkciju koristeći Adaptive Nelder-Mead Simplex (ANMS): simpleks se pomera
// refleksijom, ekspanzijom, kontrakcijom ili skupljanjem sa koeficijentima skaliranim brojem dimenzija
// problem.Point je jedan teme početnog simpleksa i određuje dimenzionalnost
func NelderMeadAdaptive(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := nelderMeadAdaptiveMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := nelderMeadAdaptiveTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	n := float64(dimensions)

	alpha := 1.0
	beta := 1 + 2/n
	gamma := 0.75 - 1/(2*n)
	delta := 1 - 1/n

	simplex := make([][]float64, dimensions+1)
	simplex[0] = append([]float64(nil), problem.Point...)
	for i := 0; i < dimensions; i++ {
		vertex := append([]float64(nil), problem.Point...)
		vertex[i] += nelderMeadAdaptiveInitialStep
		simplex[i+1] = vertex
	}

	steps := 0
	for ; steps < maxSteps; steps++ {

		sort.Slice(simplex, func(i, j int) bool {
			return fn.Evaluate(simplex[i]) < fn.Evaluate(simplex[j])
		})

		best := fn.Evaluate(simplex[0])
		worst := fn.Evaluate(simplex[dimensions])

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, simplex[0], best, false)
		}

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
			reflected[i] = centroid[i] + alpha*(centroid[i]-simplex[dimensions][i])
		}
		reflectedValue := fn.Evaluate(reflected)

		secondWorst := fn.Evaluate(simplex[dimensions-1])

		switch {
		case reflectedValue < best:
			expanded := make([]float64, dimensions)
			for i := range centroid {
				expanded[i] = centroid[i] + beta*(reflected[i]-centroid[i])
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
				contracted[i] = centroid[i] + gamma*(simplex[dimensions][i]-centroid[i])
			}
			if fn.Evaluate(contracted) < worst {
				simplex[dimensions] = contracted
			} else {
				for v := 1; v <= dimensions; v++ {
					for i := range simplex[v] {
						simplex[v][i] = simplex[0][i] + delta*(simplex[v][i]-simplex[0][i])
					}
				}
			}
		}
	}

	sort.Slice(simplex, func(i, j int) bool {
		return fn.Evaluate(simplex[i]) < fn.Evaluate(simplex[j])
	})

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, simplex[0], fn.Evaluate(simplex[0]), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "nelder_mead_adaptive",
		Point:      simplex[0],
		Value:      fn.Evaluate(simplex[0]),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
