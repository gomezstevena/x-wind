      program main ! example of calls of mesh generator with segment data
      implicit none
      integer nvmax,ntmax,nbmax
c     nvmax   - maximum number of mesh nodes
c     ntmax   - maximum number of mesh triangles
c     nbmax   - maximum number of boundary edges
      parameter(nvmax=150000,ntmax=2*nvmax,nbmax=10000)

c mesh generator data specifying domain via in the segment format
      double precision vbr(2,nbmax)
      integer          Nbr

      integer          nv,nt,nb
      double precision vrt(2,nvmax)
      integer          labelB(nbmax), labelT(ntmax)
      integer          tri(3,ntmax), bnd(2,nbmax)
c ... AFT2D library function
      Integer   aft2dfront   
      EXTERNAL  aft2dfront   

      integer          i,j,dummy,ierr

C Read input file that contains coordinates of boundary points
      Read(*,*) Nbr                   
      if (Nbr > nbmax) stop 'Nbr too large'
      Do i = 1, Nbr
         Read(*,*) (vbr(j,i),j=1,2)
      End do

C Generate a mesh  starting  from boundary mesh
      ierr=aft2dfront(
     &           0, dummy, Nbr, vbr,       ! segment data
     &           nv, vrt,                  ! mesh data on output
     &           nt, tri, labelT,
     &           nb, bnd, labelB)
      If (ierr.ne.0) stop ' error in function aft2dfront'   

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
