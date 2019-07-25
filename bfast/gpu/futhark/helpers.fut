
-- | implementation is in this entry point
--   the outer map is distributed directly
entry remove_nans [m][n][p] (nan_value: i16) (images : [m][n][p]i16) = 
	map (\block -> 
			map (\row -> 
					map (\el -> if el == nan_value then f32.nan else f32.i16 el
						) row
				) block
		) images