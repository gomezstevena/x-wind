C =====================================================================
      Program Main
C =====================================================================
      implicit none

c Maximum number of nodes, elements and boundary edges
      Integer   nvmax, nbmax, ntmax
      Parameter(nvmax= 20000, nbmax= 10000, ntmax= 2 * nvmax)

c Available memory 
      Integer   MaxWr, MaxWi
      Parameter(MaxWr = 2 000 000, MaxWi = 3 000 000)

C =====================================================================
C group (M)
      Integer  nv, nvfix, labelv(nvmax), fixedV(nvmax)
      Real*8   vrt(2, nvmax)

      Integer  nb, nbfix, bnd(2, nbmax), labelB(nbmax), fixedB(nbmax)

      Integer  nc, labelC(nbmax)
      Real*8   Crv(2, nbmax)
      EXTERNAL ANI_CrvFunction

      Integer  nt, ntfix, tri(3, ntmax), labelT(ntmax), fixedT(ntmax)

C group (CONTROL)
      Integer  control(6)
      Real*8   Metric(3, nvmax), Quality

C group (W)
      Real*8  rW(MaxWr)
      Integer iW(MaxWi)


C LOCAL VARIABLES
      Integer       i, j, nEStar, iERR
      Real*8        x, y

C =====================================================================
c ... load the initial mesh. The extension must be .ani

c     nEStar is desired number of triangles
      Read (*,*) nEStar, nv, nt, nb
      Do i = 1, nv
         Read(*,*) (vrt(j,i),j=1,2)
         labelv(i) = 0
      End do
      Do i = 1, nt
         Read(*,*) (tri(j,i),j=1,3)
         labelT(i) = 1
      End do
      Do i = 1, nb
         Read(*,*) (bnd(j,i),j=1,2)
         labelB(i) = i
      End do
      Do i = 1, nv
         Read(*,*) (Metric(j,i),j=1,3)
      End do

      nvfix = 0
      nbfix = 0
      ntfix = 0
      nc = 0

c ... generate adaptive mesh
      control(1) = nEStar / 10     !  MaxSkipE
      control(2) = 15000           !  MaxQItr
      control(3) = 1               !  status
      control(4) = 1               !  flagAuto
      control(5) = 1       !  iPrint:   minimal level of output information
      control(6) = 0       !  iErrMesgt: only critical termination allowed

      Quality = 0.8D0      !  request shape-regular triangles in metric

      Call mbaNodal(
     &      nv, nvfix, nvmax, vrt, labelv, fixedV,
     &      nb, nbfix, nbmax, bnd, labelB, fixedB,
     &      nc, Crv, labelC, ANI_CrvFunction,
     &      nt, ntfix, ntmax, tri, labelT, fixedT,
     &      nEStar, Quality, control, Metric,
     &      MaxWr, MaxWi, rW, iW, iERR)

      write (*,*) "Error ", iERR

      write (*,*) "=== Output Data Starts Here 09887654321 ===\n"

      Write (*,*) nv, nt, nb
      Do i = 1, nv
         Write(*,*) (vrt(j,i),j=1,2)
      End do
      Do i = 1, nt
         Write(*,*) (tri(j,i),j=1,3)
      End do
      Do i = 1, nb
         Write(*,*) (bnd(j,i),j=1,2)
      End do

      Stop 
      End

