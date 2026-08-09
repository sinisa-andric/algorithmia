package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	directSearchMaxSteps  = 500
	directSearchTolerance = 1e-6
	directSearchRange     = 15.0
)

type directRectangle struct {
	lower []float64
	upper []float64
}

func (r directRectangle) center() []float64 {
	c := make([]float64, len(r.lower))
	for i := range c {
		c[i] = (r.lower[i] + r.upper[i]) / 2
	}
	return c
}

// DirectSearch minimizuje sphere funkciju koristeći pojednostavljeni Dividing Rectangles (DIRECT) metod: ponavljano
// deli pravougaonik čiji je centar trenutno najbolji na tri dela duž njegove najduže dimenzije
// problem.Point samo određuje dimenzionalnost
func DirectSearch(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := directSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := directSearchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	searchRange := directSearchRange
	if v, ok := problem.Payload["range"].(float64); ok {
		searchRange = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	initial := directRectangle{
		lower: make([]float64, dimensions),
		upper: make([]float64, dimensions),
	}
	for d := 0; d < dimensions; d++ {
		initial.lower[d] = -searchRange
		initial.upper[d] = searchRange
	}

	rectangles := []directRectangle{initial}

	best := initial.center()
	bestValue := fn.Evaluate(best)

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		bestIndex := 0
		bestCenterValue := fn.Evaluate(rectangles[0].center())
		for i := 1; i < len(rectangles); i++ {
			if value := fn.Evaluate(rectangles[i].center()); value < bestCenterValue {
				bestCenterValue = value
				bestIndex = i
			}
		}

		if bestCenterValue < bestValue {
			bestValue = bestCenterValue
			best = rectangles[bestIndex].center()
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}

		target := rectangles[bestIndex]

		longest := 0
		longestWidth := target.upper[0] - target.lower[0]
		for d := 1; d < dimensions; d++ {
			width := target.upper[d] - target.lower[d]
			if width > longestWidth {
				longestWidth = width
				longest = d
			}
		}

		third := longestWidth / 3

		partA := directRectangle{lower: append([]float64(nil), target.lower...), upper: append([]float64(nil), target.upper...)}
		partA.upper[longest] = target.lower[longest] + third

		partB := directRectangle{lower: append([]float64(nil), target.lower...), upper: append([]float64(nil), target.upper...)}
		partB.lower[longest] = target.lower[longest] + third
		partB.upper[longest] = target.lower[longest] + 2*third

		partC := directRectangle{lower: append([]float64(nil), target.lower...), upper: append([]float64(nil), target.upper...)}
		partC.lower[longest] = target.lower[longest] + 2*third

		rectangles = append(rectangles[:bestIndex], rectangles[bestIndex+1:]...)
		rectangles = append(rectangles, partA, partB, partC)
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "direct_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
