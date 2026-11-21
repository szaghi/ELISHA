! the embers at opposite ends of a diameter, turning by pi / NFRAME per frame
ember(:,1) = C + ORBIT * [cos(PI * frame / NFRAME), sin(PI * frame / NFRAME), 0._R_P]
ember(:,2) = 2 * C - ember(:,1)
